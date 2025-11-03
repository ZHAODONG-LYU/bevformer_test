#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 CUDA kernel 中添加 debug 输出，验证是否使用了修改后的代码
"""

import os
from datetime import datetime

def add_debug_to_cuda():
    cuda_file = "/usr/local/lib/python3.8/dist-packages/mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh"
    
    if not os.path.exists(cuda_file):
        print(f"❌ 错误：找不到文件 {cuda_file}")
        return False
    
    # 备份
    backup_file = f"{cuda_file}.before_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"📦 备份到: {backup_file}")
    
    with open(cuda_file, 'r') as f:
        content = f.read()
    
    with open(backup_file, 'w') as f:
        f.write(content)
    
    # 查找 ms_deform_attn_im2col_nearest 函数
    lines = content.split('\n')
    
    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # 在函数开始后添加 printf
        if '__device__ scalar_t ms_deform_attn_im2col_nearest' in line:
            # 找到函数体开始的 {
            for j in range(i, min(i+10, len(lines))):
                if '{' in lines[j]:
                    # 在 { 后面添加 debug 输出
                    indent = '  '
                    debug_code = f'''{indent}// ========== DEBUG: 使用最近邻插值 ==========
{indent}if (threadIdx.x == 0 && blockIdx.x == 0) {{
{indent}  printf("[CUDA DEBUG] ✓✓✓ 使用最近邻插值（单点采样）！h=%.2f, w=%.2f\\n", h, w);
{indent}}}'''
                    
                    # 插入到函数体开始
                    new_lines.append(debug_code)
                    modified = True
                    print(f"✓ 在第 {i+1} 行的函数中添加了 debug 输出")
                    break
            break
    
    if not modified:
        print("❌ 未找到 ms_deform_attn_im2col_nearest 函数")
        return False
    
    # 写回文件
    new_content = '\n'.join(new_lines)
    with open(cuda_file, 'w') as f:
        f.write(new_content)
    
    print("✓ Debug 代码已添加")
    print()
    print("📋 下一步:")
    print("   1. 直接运行测试（会自动 JIT 重新编译）:")
    print("      ./tools/dist_test.sh \\")
    print("          projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \\")
    print("          ckpts/bevformerV2-t1-24.pth \\")
    print("          1 --eval bbox | head -100")
    print()
    print("   2. 查看输出，应该能看到:")
    print("      [CUDA DEBUG] ✓✓✓ 使用最近邻插值（单点采样）！")
    print()
    print("   3. 如果看到 debug 输出 → 证明用了新代码")
    print("      如果看不到 → 说明还是旧代码")
    print()
    
    return True

if __name__ == '__main__':
    print("=" * 70)
    print("添加 CUDA Debug 输出")
    print("=" * 70)
    print()
    
    add_debug_to_cuda()

