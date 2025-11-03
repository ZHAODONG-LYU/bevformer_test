#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Profiling utility for BEVFormer encoders.

This script runs a configurable number of inference iterations with
torch.profiler instrumentation to break down the runtime cost of the BEVFormer
encoder, focusing on Spatial Cross Attention (SCA), Temporal Cross Attention
(TCA), Feed Forward Networks (FFN), and the underlying multi-scale deformable
attention (MSDA) CUDA kernels.

It additionally captures the tensors associated with a full SCA MSDA call and
measures the standalone execution time (and simulated data movement cost) of
that CUDA kernel.

Example usage:

    python tools/profile_bevformer.py \
        projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
        ckpts/bevformerV2-t1-24.pth \
        --iters 1 --warmup-iters 1 --target-query 9662

Outputs are printed to stdout in JSON format for downstream analysis and an
optional Chrome trace can be exported for inspection with e.g. Nsight Systems.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile, record_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


@dataclass
class MSDACallInputs:
    """Container for tensors needed to replay a full MSDA kernel call."""

    value: torch.Tensor
    sampling_locations: torch.Tensor
    attention_weights: torch.Tensor
    spatial_shapes: torch.Tensor
    level_start_index: torch.Tensor
    im2col_step: int
    module_path: Optional[str]


@dataclass
class ProfilingState:
    """Holds transient profiling state shared across monkey patches."""

    target_query: Optional[int]
    target_sample_idx: int = 0
    enabled: bool = True
    current_sca_context: Optional[Dict[str, Any]] = None
    single_call_inputs: Optional[MSDACallInputs] = None
    single_call_measurement_ms: Optional[float] = None
    msda_inputs: Dict[str, MSDACallInputs] = field(default_factory=dict)
    msda_call_metadata: List[Dict[str, Any]] = field(default_factory=list)
    collecting: bool = False

    def reset_iteration_state(self):
        self.current_sca_context = None


profile_state: ProfilingState = ProfilingState(target_query=None)


def _wrap_forward_with_record_function(module_cls, label: str):
    """Wrap module forward with torch.profiler record_function."""

    if getattr(module_cls, "__profile_wrapped__", False):
        return

    original_forward = module_cls.forward

    def wrapped_forward(self, *args, **kwargs):  # type: ignore[override]
        with record_function(label):
            return original_forward(self, *args, **kwargs)

    module_cls.forward = wrapped_forward  # type: ignore[assignment]
    module_cls.__profile_wrapped__ = True  # type: ignore[attr-defined]


def _wrap_module_with_record_function(model: torch.nn.Module, module_path: str, label: str) -> bool:
    modules = dict(model.named_modules())
    module = modules.get(module_path)
    if module is None:
        return False

    if getattr(module, "__profile_wrapped__", False) and getattr(module, "__profile_label__", None) == label:
        return True

    original_forward = module.forward

    def wrapped_forward(*args, **kwargs):  # type: ignore[override]
        with record_function(label):
            return original_forward(*args, **kwargs)

    module.forward = wrapped_forward  # type: ignore[assignment]
    module.__profile_wrapped__ = True  # type: ignore[attr-defined]
    module.__profile_label__ = label  # type: ignore[attr-defined]
    return True


def _patch_spatial_cross_attention(target_query: Optional[int]):
    """Inject lightweight bookkeeping into SpatialCrossAttention."""

    from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import (  # noqa: E501
        SpatialCrossAttention,
    )

    if getattr(SpatialCrossAttention, "__profile_patched__", False):
        return

    original_forward = SpatialCrossAttention.forward

    def forward_with_context(self, *args, **kwargs):  # type: ignore[override]
        if not profile_state.enabled or target_query is None:
            with record_function("SCA"):
                return original_forward(self, *args, **kwargs)

        query = args[0] if args else kwargs.get("query")
        bev_mask = kwargs.get("bev_mask")

        context_dict = None
        if isinstance(query, torch.Tensor) and bev_mask is not None:
            indexes = []
            try:
                for mask_per_camera in bev_mask:
                    selected = mask_per_camera[0].sum(-1).nonzero(as_tuple=False).squeeze(-1)
                    indexes.append(selected)
            except Exception:
                indexes = []

            if indexes:
                context_dict = {
                    "indexes": indexes,
                    "num_cams": self.num_cams,
                    "batch_size": query.size(0),
                    "module": self,
                    "module_path": getattr(self, "_module_path", None),
                }

        profile_state.current_sca_context = context_dict

        try:
            with record_function("SCA"):
                return original_forward(self, *args, **kwargs)
        finally:
            profile_state.current_sca_context = None

    SpatialCrossAttention.forward = forward_with_context  # type: ignore[assignment]
    SpatialCrossAttention.__profile_patched__ = True  # type: ignore[attr-defined]


def _patch_msda_functions():
    """Wrap MSDA custom autograd functions for profiling and data capture."""

    from projects.mmdet3d_plugin.bevformer.modules import multi_scale_deformable_attn_function as msda_fn  # noqa: E501

    for cls_name in ["MultiScaleDeformableAttnFunction_fp32", "MultiScaleDeformableAttnFunction_fp16"]:
        cls = getattr(msda_fn, cls_name)
        if getattr(cls, "__profile_patched__", False):
            continue

        original_forward = cls.forward  # type: ignore[attr-defined]

        def forward_with_capture(*args, **kwargs):
            value = args[1]
            spatial_shapes = args[2]
            level_start_index = args[3]
            sampling_locations = args[4]
            attention_weights = args[5]
            im2col_step = args[6]

            with record_function("MSDA"):
                output = original_forward(*args, **kwargs)

            context = profile_state.current_sca_context
            module_path = None
            if context is not None:
                module_path = context.get("module_path")

            if (
                profile_state.enabled
                and context is not None
                and profile_state.collecting
            ):
                capture_required = False
                indexes = context.get("indexes", [])
                if profile_state.target_query is None or not indexes:
                    capture_required = True
                else:
                    target = profile_state.target_query
                    for idx_tensor in indexes:
                        if ((idx_tensor == target).any()).item():
                            capture_required = True
                            break

                if capture_required and profile_state.single_call_inputs is None:
                    profile_state.single_call_inputs = MSDACallInputs(
                        value=value.detach().clone(),
                        sampling_locations=sampling_locations.detach().clone(),
                        attention_weights=attention_weights.detach().clone(),
                        spatial_shapes=spatial_shapes.detach().clone(),
                        level_start_index=level_start_index.detach().clone(),
                        im2col_step=int(im2col_step),
                        module_path=module_path,
                    )
                if module_path and module_path not in profile_state.msda_inputs:
                    profile_state.msda_inputs[module_path] = MSDACallInputs(
                        value=value.detach().clone(),
                        sampling_locations=sampling_locations.detach().clone(),
                        attention_weights=attention_weights.detach().clone(),
                        spatial_shapes=spatial_shapes.detach().clone(),
                        level_start_index=level_start_index.detach().clone(),
                        im2col_step=int(im2col_step),
                        module_path=module_path,
                    )

            # Lightweight metadata for every MSDA invocation
            if profile_state.collecting and context is not None:
                profile_state.msda_call_metadata.append(
                    {
                        "num_batches": value.shape[0],
                        "num_queries": sampling_locations.shape[1],
                        "num_heads": sampling_locations.shape[2],
                        "head_dim": value.shape[-1],
                        "num_levels": sampling_locations.shape[3],
                        "num_points": sampling_locations.shape[4],
                        "module_path": module_path,
                    }
                )

            return output

        cls.forward = forward_with_capture  # type: ignore[assignment]
        cls.__profile_patched__ = True  # type: ignore[attr-defined]


def _patch_temporal_self_attention():
    from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import (  # noqa: E501
        TemporalSelfAttention,
    )

    _wrap_forward_with_record_function(TemporalSelfAttention, "TCA")


def _patch_ffn():
    from mmcv.cnn.bricks.transformer import FFN

    _wrap_forward_with_record_function(FFN, "FFN")


def _apply_module_path_metadata(model: torch.nn.Module):
    for name, module in model.named_modules():
        setattr(module, "_module_path", name)


def _measure_full_msda(inputs: MSDACallInputs, repeat: int = 10) -> Dict[str, Any]:
    from projects.mmdet3d_plugin.bevformer.modules import multi_scale_deformable_attn_function as msda_fn  # noqa: E501

    repeat = max(repeat, 1)
    io_samples: List[float] = []
    kernel_samples: List[float] = []
    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            io_start = torch.cuda.Event(enable_timing=True)
            io_end = torch.cuda.Event(enable_timing=True)
            kernel_start = torch.cuda.Event(enable_timing=True)
            kernel_end = torch.cuda.Event(enable_timing=True)

            io_start.record()
            value = inputs.value.clone()
            sampling_locations = inputs.sampling_locations.clone()
            attention_weights = inputs.attention_weights.clone()
            io_end.record()
            torch.cuda.synchronize()
            io_samples.append(float(io_start.elapsed_time(io_end)))

            kernel_start.record()
            _ = msda_fn.ext_module.ms_deform_attn_forward(
                value,
                inputs.spatial_shapes,
                inputs.level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step=inputs.im2col_step,
            )
            kernel_end.record()
            torch.cuda.synchronize()
            kernel_samples.append(float(kernel_start.elapsed_time(kernel_end)))

    def _stats(samples: List[float]) -> Dict[str, Any]:
        trimmed = samples[1:] if len(samples) > 1 else samples
        mean_ms = float(sum(trimmed) / len(trimmed)) if trimmed else float("nan")
        min_ms = float(min(trimmed)) if trimmed else float("nan")
        max_ms = float(max(trimmed)) if trimmed else float("nan")
        return {
            "samples_ms": samples,
            "mean_ms": mean_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
        }

    total_samples = [i + k for i, k in zip(io_samples, kernel_samples)]

    return {
        "repeat": repeat,
        "io": _stats(io_samples),
        "kernel": _stats(kernel_samples),
        "total": _stats(total_samples),
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile BEVFormer encoder components")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run profiling on")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--warmup-iters", type=int, default=1, help="Number of warmup iterations before profiling")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations to profile")
    parser.add_argument("--target-query", type=int, default=9662, help="BEV query index to capture for MSDA micro-benchmark")
    parser.add_argument("--target-sample", type=int, default=0, help="Batch sample index to inspect for the target query")
    parser.add_argument("--export-trace", type=str, default=None, help="Optional path to export Chrome trace (JSON)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save summary JSON")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of dataloader workers (defaults to config value)",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to use for profiling",
    )
    parser.add_argument(
        "--msda-replay-iters",
        type=int,
        default=10,
        help="Number of times to replay the captured MSDA kernel for measurement",
    )
    parser.add_argument(
        "--pie-output",
        type=str,
        default=None,
        help="Optional path to save a pie chart of BEV encoder time breakdown",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for profiling")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model if hasattr(cfg, "model") else None
    if model_cfg is not None:
        model_cfg.setdefault("pretrained", None)

    if not hasattr(cfg, "data") or cfg.data is None:
        raise ValueError("Config must contain a 'data' section for profiling")

    data_section = cfg.data
    data_section.setdefault("workers_per_gpu", 4)

    if args.num_workers is not None:
        data_section["workers_per_gpu"] = args.num_workers

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    dataset_cfg = data_section[args.data_split]
    if isinstance(dataset_cfg, dict):
        dataset_cfg = dataset_cfg.copy()
        dataset_cfg.setdefault("test_mode", True)
        for removable_key in ["samples_per_gpu", "workers_per_gpu", "persistent_workers"]:
            dataset_cfg.pop(removable_key, None)
    else:
        raise TypeError("Dataset config must be a dict for profiling")

    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=data_section.get("workers_per_gpu", 4),
        num_gpus=1,
        dist=False,
        shuffle=False,
    )

    from mmdet3d.models import build_model

    model = build_model(model_cfg, test_cfg=cfg.get("test_cfg"))
    if args.checkpoint.lower() != "none":
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model = model.cuda(device)
    model.eval()

    mm_model = MMDataParallel(model, device_ids=[device.index if device.index is not None else 0])
    mm_model.eval()

    _apply_module_path_metadata(mm_model.module)

    profile_state.target_query = args.target_query
    profile_state.target_sample_idx = args.target_sample
    profile_state.enabled = args.target_query is not None

    _patch_spatial_cross_attention(profile_state.target_query)
    _patch_temporal_self_attention()
    _patch_ffn()
    _patch_msda_functions()

    _wrap_module_with_record_function(mm_model.module, "pts_bbox_head.transformer.encoder", "BEVEncoder")
    for layer_idx in range(0, 12):
        layer_path = f"pts_bbox_head.transformer.encoder.layers.{layer_idx}"
        if not _wrap_module_with_record_function(mm_model.module, layer_path, f"BEVLayer_{layer_idx}"):
            break

    # Warmup iterations
    data_iter = iter(data_loader)
    profile_state.single_call_inputs = None
    profile_state.single_call_measurement_ms = None
    profile_state.msda_inputs = {}
    profile_state.msda_call_metadata = []
    def run_inference(batch):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                mm_model(return_loss=False, rescale=True, **batch)

    profile_state.collecting = False
    for _ in range(args.warmup_iters):
        data = next(data_iter)
        profile_state.reset_iteration_state()
        run_inference(data)
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    profiled_steps = 0
    with profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for _ in range(args.iters):
            data = next(data_iter)
            profile_state.reset_iteration_state()
            profile_state.collecting = True
            run_inference(data)
            torch.cuda.synchronize()
            profiled_steps += 1
            prof.step()
    profile_state.collecting = False

    if args.export_trace:
        export_path = Path(args.export_trace)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(export_path))

    # Collect profiler statistics
    summary: Dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "warmup_iters": args.warmup_iters,
        "profile_iters": profiled_steps,
        "target_query": args.target_query,
        "metrics": {},
        "msda_calls": profile_state.msda_call_metadata,
    }

    key_avgs = prof.key_averages()
    total_cuda_time = sum(item.self_cuda_time_total for item in key_avgs)
    summary["metrics"]["total_cuda_time_ms"] = total_cuda_time / 1000.0

    def extract(label: str) -> Dict[str, float]:
        for item in key_avgs:
            if item.key == label:
                return {
                    "cuda_time_total_ms": item.cuda_time_total / 1000.0,
                    "self_cuda_time_ms": item.self_cuda_time_total / 1000.0,
                    "cpu_time_total_ms": item.cpu_time_total / 1000.0,
                    "count": item.count,
                }
        return {
            "cuda_time_total_ms": 0.0,
            "self_cuda_time_ms": 0.0,
            "cpu_time_total_ms": 0.0,
            "count": 0,
        }

    summary["metrics"]["SCA"] = extract("SCA")
    summary["metrics"]["TCA"] = extract("TCA")
    summary["metrics"]["FFN"] = extract("FFN")
    summary["metrics"]["MSDA"] = extract("MSDA")
    summary["metrics"]["BEVEncoder"] = extract("BEVEncoder")

    bev_layer_metrics: List[Dict[str, Any]] = []
    for item in key_avgs:
        if item.key.startswith("BEVLayer_"):
            bev_layer_metrics.append(
                {
                    "layer": item.key,
                    "cuda_time_total_ms": item.cuda_time_total / 1000.0,
                    "self_cuda_time_ms": item.self_cuda_time_total / 1000.0,
                    "cpu_time_total_ms": item.cpu_time_total / 1000.0,
                    "count": item.count,
                }
            )
    bev_layer_metrics.sort(key=lambda x: x["layer"])
    if bev_layer_metrics:
        summary["metrics"]["BEVEncoder_layers"] = bev_layer_metrics

    bev_total = summary["metrics"]["BEVEncoder"].get("cuda_time_total_ms", 0.0)
    if bev_total > 0:
        bev_sca = summary["metrics"]["SCA"].get("cuda_time_total_ms", 0.0)
        bev_tca = summary["metrics"]["TCA"].get("cuda_time_total_ms", 0.0)
        bev_ffn = summary["metrics"]["FFN"].get("cuda_time_total_ms", 0.0)
        bev_msda = summary["metrics"]["MSDA"].get("cuda_time_total_ms", 0.0)
        bev_other = max(bev_total - (bev_sca + bev_tca + bev_ffn), 0.0)
        summary["metrics"]["BEVEncoder_breakdown"] = {
            "SCA_ms": bev_sca,
            "TCA_ms": bev_tca,
            "FFN_ms": bev_ffn,
            "Other_ms": bev_other,
            "MSDA_ms": bev_msda,
            "SCA_pct": bev_sca / bev_total,
            "TCA_pct": bev_tca / bev_total,
            "FFN_pct": bev_ffn / bev_total,
            "Other_pct": bev_other / bev_total,
            "MSDA_pct_of_BEV": bev_msda / bev_total,
        }

    if profile_state.msda_call_metadata:
        summary["msda_calls"] = profile_state.msda_call_metadata
        aggregate: Dict[str, Dict[str, Any]] = {}
        for call in profile_state.msda_call_metadata:
            key = call.get("module_path", "unknown")
            entry = aggregate.setdefault(
                key,
                {
                    "module_path": key,
                    "count": 0,
                    "num_batches": call["num_batches"],
                    "num_queries": call["num_queries"],
                    "num_heads": call["num_heads"],
                    "head_dim": call["head_dim"],
                    "num_levels": call["num_levels"],
                    "num_points": call["num_points"],
                },
            )
            entry["count"] += 1
        summary["msda_call_aggregate"] = sorted(
            aggregate.values(), key=lambda x: x["module_path"]
        )
    else:
        summary["msda_calls"] = []
        summary["msda_call_aggregate"] = []

    mmcv_cuda = sum(
        item.self_cuda_time_total
        for item in key_avgs
        if "mmcv" in item.key.lower() or "mm" in item.key.lower()
    )
    summary["metrics"]["mmcv_estimated_overhead_ms"] = mmcv_cuda / 1000.0

    if profile_state.single_call_inputs is not None:
        msda_stats = _measure_full_msda(
            profile_state.single_call_inputs, repeat=args.msda_replay_iters
        )
        profile_state.single_call_measurement_ms = msda_stats.get("total", {}).get("mean_ms")
        inputs = profile_state.single_call_inputs
        summary["msda_full_call"] = {
            "module_path": inputs.module_path,
            "num_batches": inputs.value.shape[0],
            "num_queries": inputs.sampling_locations.shape[1],
            "num_heads": inputs.sampling_locations.shape[2],
            "head_dim": inputs.value.shape[-1],
            "num_levels": inputs.sampling_locations.shape[3],
            "num_points": inputs.sampling_locations.shape[4],
            **msda_stats,
        }
    else:
        summary["msda_full_call"] = {
            "error": "SCA call not captured; ensure target query is present"
        }

    if profile_state.msda_inputs:
        msda_layer_stats: List[Dict[str, Any]] = []
        total_io = 0.0
        total_kernel = 0.0
        total_total = 0.0
        for module_path in sorted(profile_state.msda_inputs.keys()):
            inputs = profile_state.msda_inputs[module_path]
            stats = _measure_full_msda(inputs, repeat=args.msda_replay_iters)
            io_mean = stats.get("io", {}).get("mean_ms", 0.0)
            kernel_mean = stats.get("kernel", {}).get("mean_ms", 0.0)
            total_mean = stats.get("total", {}).get("mean_ms", 0.0)
            total_io += io_mean
            total_kernel += kernel_mean
            total_total += total_mean
            msda_layer_stats.append(
                {
                    "module_path": module_path,
                    "num_batches": inputs.value.shape[0],
                    "num_queries": inputs.sampling_locations.shape[1],
                    "num_heads": inputs.sampling_locations.shape[2],
                    "head_dim": inputs.value.shape[-1],
                    "num_levels": inputs.sampling_locations.shape[3],
                    "num_points": inputs.sampling_locations.shape[4],
                    **stats,
                }
            )
        summary["msda_full_call_per_layer"] = msda_layer_stats
        summary["msda_full_call_totals"] = {
            "io_ms": total_io,
            "kernel_ms": total_kernel,
            "total_ms": total_total,
        }
    else:
        summary["msda_full_call_per_layer"] = []

    if args.pie_output and "BEVEncoder_breakdown" in summary["metrics"]:
        pie_path = Path(args.pie_output)
        pie_path.parent.mkdir(parents=True, exist_ok=True)
        breakdown = summary["metrics"]["BEVEncoder_breakdown"]
        labels = ["SCA", "TCA", "FFN", "Other"]
        sizes = [
            breakdown.get("SCA_ms", 0.0),
            breakdown.get("TCA_ms", 0.0),
            breakdown.get("FFN_ms", 0.0),
            breakdown.get("Other_ms", 0.0),
        ]
        if sum(sizes) > 0:
            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.title("BEV Encoder Time Breakdown")
            plt.tight_layout()
            plt.savefig(pie_path)
            plt.close()
            summary.setdefault("artifacts", {})["pie_chart"] = str(pie_path)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


