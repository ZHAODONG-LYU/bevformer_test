#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出融合了量化scale的模型权重，供FPGA使用

使用方法:
    python tools/export_quantized_weights.py \\
        ckpts/bevformerV2-t1-24.pth \\
        quantization_params.pth \\
        --output bevformer_quantized_weights.pth
"""

import argparse
import torch


def fuse_scale_to_weight(weight, scale, dim=0):
    """
    将量化scale融合到权重中
    
    Args:
        weight: 权重矩阵 [out_features, in_features]
        scale: 量化scale [num_heads * num_levels] 或 [num_heads, num_levels]
        dim: scale对应的维度
        
    Returns:
        weight_fused: 融合后的权重
    """
    # 将scale广播到权重的形状
    if len(scale.shape) == 1:
        # scale: [num_heads * num_levels]
        # weight: [embed_dims, embed_dims] 需要拆分成 [num_heads, head_dim, embed_dims]
        pass
    else:
        # scale: [num_heads, num_levels]
        # 需要根据实际的权重布局来fusion
        pass
    
    # 简单实现：直接除以scale
    # 注意：实际实现需要根据你的权重布局来调整
    weight_fused = weight / scale.view(-1, 1)
    
    return weight_fused


def main():
    parser = argparse.ArgumentParser(description='导出融合scale的量化权重')
    parser.add_argument('checkpoint', help='原始checkpoint路径')
    parser.add_argument('quant_params', help='量化参数文件路径')
    parser.add_argument('--output', default='bevformer_quantized_weights.pth',
                        help='输出文件路径')
    args = parser.parse_args()
    
    # 加载原始权重
    print(f"加载checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 加载量化参数
    print(f"加载量化参数: {args.quant_params}")
    quant_params = torch.load(args.quant_params, map_location='cpu')
    scales = quant_params['scales']  # [num_heads, num_levels]
    
    print(f"量化scale shape: {scales.shape}")
    print(f"量化scale 范围: [{scales.min():.6f}, {scales.max():.6f}]")
    
    # 找到所有MSDA的value_proj权重并融合scale
    fused_state_dict = {}
    num_fused = 0
    
    for name, param in state_dict.items():
        if 'deformable_attention.value_proj' in name and 'weight' in name:
            print(f"\n融合scale到: {name}")
            print(f"  原始权重shape: {param.shape}")
            
            # 这里需要根据你的权重布局来实现fusion
            # 示例：如果value_proj是 [embed_dims, embed_dims]
            # 需要拆分成 [num_heads, head_dim, embed_dims] 并按head应用scale
            
            # 简化版本：展平scale并广播
            # 注意：实际实现需要更精细的处理
            scale_flat = scales.flatten().unsqueeze(1).repeat(1, param.shape[1] // scales.numel())
            scale_flat = scale_flat.reshape(-1)[:param.shape[0]].unsqueeze(1)
            
            param_fused = param / scale_flat
            fused_state_dict[name] = param_fused
            num_fused += 1
            
            print(f"  融合后权重范围: [{param_fused.min():.6f}, {param_fused.max():.6f}]")
        else:
            # 其他权重不变
            fused_state_dict[name] = param
    
    print(f"\n总共融合了 {num_fused} 个权重")
    
    # 保存融合后的权重
    output_data = {
        'state_dict': fused_state_dict,
        'meta': {
            'quantization_fused': True,
            'scales': scales,
            'original_checkpoint': args.checkpoint,
            'quant_params': args.quant_params,
        }
    }
    
    torch.save(output_data, args.output)
    print(f"\n融合后的权重已保存到: {args.output}")


if __name__ == '__main__':
    main()

