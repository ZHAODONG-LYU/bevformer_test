#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
收集 BEVFormer MSDA value feature map 的校准数据，用于计算 int8 量化参数

使用方法：
    python tools/collect_quantization_calibration.py \\
        projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \\
        ckpts/bevformerV2-t1-24.pth \\
        --calib-samples 100 \\
        --output quantization_params.pth
"""

import argparse
import os
import sys
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


class CalibrationCollector:
    """收集 MSDA value feature map 的统计信息"""
    
    def __init__(self):
        self.enabled = False
        self.collecting = False
        # 存储每个 (head, level) 的 value 绝对值最大值
        self.value_max_per_head_level = defaultdict(lambda: 0.0)
        # 存储所有 value 样本，用于计算分位数（可选）
        self.value_samples = defaultdict(list)
        self.num_samples_collected = 0
        self.max_samples_to_store = 1000  # 每个 (head, level) 最多存 1000 个样本值
        
    def reset(self):
        self.value_max_per_head_level.clear()
        self.value_samples.clear()
        self.num_samples_collected = 0
        
    def collect_value(self, value, spatial_shapes, module_path=None):
        """
        收集 value feature map 的统计信息
        
        Args:
            value: [B, N, num_heads, head_dim]  (B=batch*num_cams, N=num_queries)
            spatial_shapes: [num_levels, 2]  每个 level 的空间尺寸
            module_path: 模块路径（用于区分不同的 SCA）
        """
        if not self.enabled or not self.collecting:
            return
        
        # 打印第一次的 shape 信息
        if self.num_samples_collected == 0:
            print(f"[Calibration Debug] value shape: {value.shape}")
            print(f"[Calibration Debug] spatial_shapes: {spatial_shapes}")
            
        B, N, num_heads, head_dim = value.shape
        num_levels = spatial_shapes.shape[0]
        
        # 将 value 按 level 分组（需要根据 spatial_shapes 计算每个 level 的 query 数量）
        # 注意：这里假设 value 已经是 flatten 后按 level 顺序排列的
        # 如果你的实现不同，需要根据实际情况调整
        
        level_start_idx = [0]
        for lvl in range(num_levels):
            h, w = spatial_shapes[lvl]
            level_start_idx.append(level_start_idx[-1] + int(h * w))
        
        # 对每个 (head, level) 收集统计
        for lvl in range(num_levels):
            start = level_start_idx[lvl]
            end = level_start_idx[lvl + 1]
            
            if start >= N or end > N:
                # 如果索引超出范围，可能是 query=9662 的情况（BEV queries）
                # 这种情况下，value 不是按 level 分的，而是整体的 BEV queries
                # 我们简化处理：对整个 value 统计
                for head_idx in range(num_heads):
                    value_head = value[:, :, head_idx, :]  # [B, N, head_dim]
                    max_val = value_head.abs().max().item()
                    
                    key = (head_idx, 0)  # 使用 level=0 作为占位符
                    self.value_max_per_head_level[key] = max(
                        self.value_max_per_head_level[key], max_val
                    )
                    
                    # 可选：收集样本用于分位数计算
                    if len(self.value_samples[key]) < self.max_samples_to_store:
                        samples = value_head.flatten().abs().cpu().numpy()
                        samples = np.random.choice(samples, min(100, len(samples)), replace=False)
                        self.value_samples[key].extend(samples.tolist())
                break
            else:
                # 按 level 分组统计
                value_level = value[:, start:end, :, :]  # [B, num_queries_level, num_heads, head_dim]
                
                for head_idx in range(num_heads):
                    value_head = value_level[:, :, head_idx, :]  # [B, num_queries_level, head_dim]
                    max_val = value_head.abs().max().item()
                    
                    key = (head_idx, lvl)
                    self.value_max_per_head_level[key] = max(
                        self.value_max_per_head_level[key], max_val
                    )
                    
                    # 可选：收集样本
                    if len(self.value_samples[key]) < self.max_samples_to_store:
                        samples = value_head.flatten().abs().cpu().numpy()
                        samples = np.random.choice(samples, min(100, len(samples)), replace=False)
                        self.value_samples[key].extend(samples.tolist())
        
        self.num_samples_collected += 1
        
    def compute_scales(self, method='max', percentile=99.9):
        """
        计算量化 scale
        
        Args:
            method: 'max' 使用最大值，'percentile' 使用分位数
            percentile: 分位数百分比（默认 99.9）
            
        Returns:
            scales: dict {(head_idx, level_idx): scale_value}
        """
        scales = {}
        
        if method == 'max':
            for key, max_val in self.value_max_per_head_level.items():
                scales[key] = max_val / 127.0
        elif method == 'percentile':
            for key, samples in self.value_samples.items():
                if len(samples) > 0:
                    pct_val = np.percentile(samples, percentile)
                    scales[key] = pct_val / 127.0
                else:
                    # 如果没有样本，使用 max
                    scales[key] = self.value_max_per_head_level.get(key, 1.0) / 127.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return scales


# 全局 collector
calibration_collector = CalibrationCollector()


def _patch_msdeformable_attention():
    """Patch MSDeformableAttention3D 来收集 value 的统计信息"""
    from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
    
    original_forward = MSDeformableAttention3D.forward
    
    def patched_forward(self, query, key=None, value=None, *args, **kwargs):
        # 调用原始 forward，但在 value_proj 之后收集统计信息
        if value is None:
            value = query
        
        # 保存原始 value_proj
        original_value_proj_forward = self.value_proj.forward
        
        def patched_value_proj_forward(v):
            # 调用原始 value_proj
            v_proj = original_value_proj_forward(v)
            
            # 收集统计信息（在 reshape 到 [bs, num_value, num_heads, head_dim] 之后）
            if calibration_collector.enabled and calibration_collector.collecting:
                bs, num_value, _ = v_proj.shape
                spatial_shapes = kwargs.get('spatial_shapes', None)
                
                # Reshape 成 [bs, num_value, num_heads, head_dim]
                v_reshaped = v_proj.view(bs, num_value, self.num_heads, -1)
                
                if spatial_shapes is not None:
                    calibration_collector.collect_value(
                        v_reshaped,
                        spatial_shapes,
                        module_path=getattr(self, '_module_path', None)
                    )
            
            return v_proj
        
        # 临时替换 value_proj.forward
        self.value_proj.forward = patched_value_proj_forward
        
        try:
            result = original_forward(self, query, key, value, *args, **kwargs)
        finally:
            # 恢复原始 value_proj.forward
            self.value_proj.forward = original_value_proj_forward
        
        return result
    
    MSDeformableAttention3D.forward = patched_forward
    print("[Calibration] Patched MSDeformableAttention3D.forward")


def parse_args():
    parser = argparse.ArgumentParser(description='收集 MSDA 量化校准数据')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='checkpoint 文件路径')
    parser.add_argument('--calib-samples', type=int, default=100,
                        help='用于校准的样本数量（默认 100）')
    parser.add_argument('--output', default='quantization_params.pth',
                        help='输出量化参数文件路径')
    parser.add_argument('--method', choices=['max', 'percentile'], default='max',
                        help='计算 scale 的方法：max 或 percentile')
    parser.add_argument('--percentile', type=float, default=99.9,
                        help='使用 percentile 方法时的分位数（默认 99.9）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--device', default='cuda:0', help='使用的设备')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 构建数据集（使用 test 集，因为 val 集会尝试加载训练标签）
    if not hasattr(cfg, 'data'):
        raise ValueError("配置文件缺少 data 部分")
    
    # 优先使用 test，如果没有则用 val
    if hasattr(cfg.data, 'test'):
        dataset_cfg = cfg.data.test.copy()
    elif hasattr(cfg.data, 'val'):
        dataset_cfg = cfg.data.val.copy()
    else:
        raise ValueError("配置文件缺少 data.test 或 data.val 部分")
    
    # 移除不兼容的参数
    for key in ['samples_per_gpu', 'workers_per_gpu', 'persistent_workers']:
        if key in dataset_cfg:
            del dataset_cfg[key]
    
    # 强制设置为 test_mode，避免加载训练标签
    dataset_cfg['test_mode'] = True
    
    dataset = build_dataset(dataset_cfg)
    
    # 构建 dataloader（单卡，batch_size=1）
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False
    )
    
    # 构建模型
    if not hasattr(cfg, 'model'):
        raise ValueError("配置文件缺少 model 部分")
    
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载 checkpoint
    if args.checkpoint and args.checkpoint != 'none':
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
    
    # 将模型移到指定设备
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # Patch MSDeformableAttention3D
    _patch_msdeformable_attention()
    
    # 开始收集校准数据
    calibration_collector.enabled = True
    calibration_collector.collecting = True
    calibration_collector.reset()
    
    print(f"[Calibration] 开始收集校准数据，使用 {args.calib_samples} 个样本...")
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= args.calib_samples:
                break
            
            # 前向传播
            result = model(return_loss=False, rescale=True, **data)
            
            if (i + 1) % 10 == 0:
                print(f"[Calibration] 已处理 {i + 1}/{args.calib_samples} 个样本")
    
    calibration_collector.collecting = False
    
    # 计算 scale
    print(f"[Calibration] 计算量化 scale（方法：{args.method}）...")
    scales = calibration_collector.compute_scales(
        method=args.method,
        percentile=args.percentile
    )
    
    # 整理成更友好的格式
    num_heads = max(k[0] for k in scales.keys()) + 1
    num_levels = max(k[1] for k in scales.keys()) + 1
    
    scale_tensor = torch.zeros(num_heads, num_levels)
    for (head_idx, level_idx), scale_val in scales.items():
        scale_tensor[head_idx, level_idx] = scale_val
    
    # 保存量化参数
    output_data = {
        'scales': scale_tensor,
        'num_heads': num_heads,
        'num_levels': num_levels,
        'method': args.method,
        'percentile': args.percentile if args.method == 'percentile' else None,
        'calib_samples': args.calib_samples,
        'config': args.config,
        'checkpoint': args.checkpoint,
    }
    
    torch.save(output_data, args.output)
    
    print(f"\n[Calibration] 量化参数已保存到: {args.output}")
    print(f"[Calibration] Scale shape: {scale_tensor.shape}")
    print(f"[Calibration] Scale 统计:")
    print(f"  - Min: {scale_tensor[scale_tensor > 0].min().item():.6f}")
    print(f"  - Max: {scale_tensor.max().item():.6f}")
    print(f"  - Mean: {scale_tensor[scale_tensor > 0].mean().item():.6f}")
    print(f"\n详细 scale 值 (head × level):")
    print(scale_tensor)


if __name__ == '__main__':
    main()

