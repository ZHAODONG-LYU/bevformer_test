#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEVFormer MSDA Int8 量化推理脚本

使用方法:
    ./tools/dist_test_quantized.sh \\
        projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \\
        ckpts/bevformerV2-t1-24.pth \\
        1 \\
        --quantization-params quantization_params.pth

说明:
    - 会在 MSDA 的 value 输入上应用 int8 量化/反量化
    - 使用 per-head, per-level 的 scale
    - 不修改模型权重，只在推理时模拟量化
"""

import argparse
import os
import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, get_dist_info, init_dist, wrap_fp16_model
from mmcv.parallel import MMDistributedDataParallel
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
from mmdet.apis import set_random_seed
import warnings
import time
import os.path as osp

# 全局量化参数
quantization_scales = None
quantization_enabled = False


def quantize_tensor(tensor, scale, debug=False):
    """
    将 FP32 tensor 量化为 int8，然后反量化回 FP32
    
    Args:
        tensor: [B, N, num_heads, head_dim] 的特征
        scale: [num_heads] 的 scale
        debug: 是否打印 debug 信息
    
    Returns:
        量化后再反量化的 FP32 tensor
    """
    # tensor: [B, N, num_heads, head_dim]
    # scale: [num_heads]
    
    # 将 scale 移到正确的设备
    if scale.device != tensor.device:
        scale = scale.to(tensor.device)
    
    # scale shape: [num_heads] -> [1, 1, num_heads, 1] 以便广播
    scale_expanded = scale.view(1, 1, -1, 1)
    
    # 量化前的统计
    if debug:
        print(f"[Quant Debug] 原始 value 范围: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        print(f"[Quant Debug] Scale 范围: [{scale.min().item():.4f}, {scale.max().item():.4f}]")
    
    # 量化: FP32 -> int8
    tensor_scaled = tensor / scale_expanded
    tensor_int8 = torch.clamp(torch.round(tensor_scaled), -127, 127)
    
    # 反量化: int8 -> FP32
    tensor_dequant = tensor_int8 * scale_expanded
    
    if debug:
        print(f"[Quant Debug] 量化后 value 范围: [{tensor_dequant.min().item():.4f}, {tensor_dequant.max().item():.4f}]")
        # 计算量化误差
        quant_error = (tensor - tensor_dequant).abs().mean().item()
        print(f"[Quant Debug] 平均量化误差: {quant_error:.6f}")
        print(f"[Quant Debug] 相对误差: {quant_error / (tensor.abs().mean().item() + 1e-8):.4%}")
    
    return tensor_dequant


def patch_msda_with_quantization():
    """Patch MSDA 的 forward 函数，在 value 上应用量化"""
    from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
    
    original_forward = MSDeformableAttention3D.forward
    
    # 用于控制 debug 输出（只在第一次调用时打印）
    quantization_debug_counter = {'count': 0}
    
    def quantized_forward(self, query, key=None, value=None, *args, **kwargs):
        if not quantization_enabled:
            return original_forward(self, query, key=key, value=value, *args, **kwargs)
        
        # 获取原始参数
        if value is None:
            value = query
        
        # 调用原始 forward，但在中间插入量化
        # 需要 hook value_proj 之后的 value
        
        # 保存原始的 value_proj.forward
        original_value_proj_forward = self.value_proj.forward
        
        def quantized_value_proj_forward(v):
            # 调用原始 value_proj
            v_proj = original_value_proj_forward(v)
            
            # 获取 spatial_shapes
            spatial_shapes = kwargs.get('spatial_shapes', None)
            
            if spatial_shapes is not None and quantization_scales is not None:
                bs, num_value, _ = v_proj.shape
                
                # Reshape 成 [bs, num_value, num_heads, head_dim]
                v_reshaped = v_proj.view(bs, num_value, self.num_heads, -1)
                
                # 按 level 拆分并量化
                num_levels = spatial_shapes.shape[0]
                level_start_idx = 0
                v_quantized_list = []
                
                for lvl_idx in range(num_levels):
                    h, w = spatial_shapes[lvl_idx].tolist()
                    level_size = h * w
                    level_end_idx = level_start_idx + level_size
                    
                    # 提取当前 level 的 value
                    v_level = v_reshaped[:, level_start_idx:level_end_idx, :, :]
                    
                    # 获取当前 level 的 scale: [num_heads]
                    scale_level = quantization_scales[:, lvl_idx]
                    
                    # 量化（只在第一次调用时打印 debug）
                    should_debug = quantization_debug_counter['count'] == 0
                    v_level_quant = quantize_tensor(v_level, scale_level, debug=should_debug)
                    v_quantized_list.append(v_level_quant)
                    
                    if should_debug:
                        print(f"[Quant Debug] Level {lvl_idx}: shape={v_level.shape}, scale={scale_level.mean().item():.4f}")
                    
                    level_start_idx = level_end_idx
                
                quantization_debug_counter['count'] += 1
                
                # 拼接回去
                v_quantized = torch.cat(v_quantized_list, dim=1)
                
                # Reshape 回 [bs, num_value, embed_dims]
                v_proj = v_quantized.reshape(bs, num_value, -1)
            
            return v_proj
        
        # 临时替换 value_proj.forward
        self.value_proj.forward = quantized_value_proj_forward
        
        try:
            result = original_forward(self, query, key, value, *args, **kwargs)
        finally:
            # 恢复原始 value_proj.forward
            self.value_proj.forward = original_value_proj_forward
        
        return result
    
    MSDeformableAttention3D.forward = quantized_forward
    print("[Quantization] Patched MSDeformableAttention3D.forward with int8 quantization")


def parse_args():
    parser = argparse.ArgumentParser(description='测试 BEVFormer 带 int8 量化')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='checkpoint 文件路径')
    parser.add_argument('--quantization-params', type=str, required=True,
                        help='量化参数文件路径 (quantization_params.pth)')
    parser.add_argument('--out', help='输出结果文件 (pickle)')
    parser.add_argument('--eval', type=str, nargs='+', 
                        help='评估指标，例如 "bbox"')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--show-dir', help='保存可视化结果的目录')
    parser.add_argument('--gpu-collect', action='store_true',
                        help='是否使用 GPU 收集结果')
    parser.add_argument('--tmpdir', help='临时目录用于收集结果')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--deterministic', action='store_true',
                        help='是否设置确定性选项')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='覆盖配置选项')
    parser.add_argument('--eval-options', nargs='+', action=DictAction,
                        help='自定义评估选项')
    parser.add_argument('--disable-quantization', action='store_true',
                        help='禁用量化（baseline 测试）')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    
    global quantization_scales, quantization_enabled
    
    # 加载量化参数
    if not args.disable_quantization:
        if not os.path.exists(args.quantization_params):
            raise FileNotFoundError(f"量化参数文件不存在: {args.quantization_params}")
        
        quant_params = torch.load(args.quantization_params, map_location='cpu')
        quantization_scales = quant_params['scales']  # [num_heads, num_levels]
        quantization_enabled = True
        
        print(f"[Quantization] 加载量化参数: {args.quantization_params}")
        print(f"[Quantization] Scale shape: {quantization_scales.shape}")
        print(f"[Quantization] Scale 范围: [{quantization_scales.min().item():.4f}, {quantization_scales.max().item():.4f}]")
    else:
        quantization_enabled = False
        print("[Quantization] 量化已禁用 (baseline mode)")
    
    # 加载配置（完全按照 test.py）
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # 导入自定义模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # 导入插件
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
    
    # CUDNN benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    cfg.model.pretrained = None
    
    # 构建数据集（完全按照 test.py）
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
    # 初始化分布式环境
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    # 设置随机种子
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    
    # 构建数据加载器
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    
    # 构建模型
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    # 应用量化 patch
    if quantization_enabled:
        patch_msda_with_quantization()
    
    # 使用分布式测试（即使单 GPU）
    if not distributed:
        raise NotImplementedError("请使用 launcher='pytorch' 运行")
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\n保存结果到: {args.out}')
            mmcv.dump(outputs, args.out)
        
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split('/')[-1].split('.')[-2], 
                                             time.ctime().replace(' ', '_').replace(':', '_'))
        
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
