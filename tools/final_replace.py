#!/usr/bin/env python3
"""最终替换脚本 - 精确替换第22-67行"""

def final_replace():
    cuda_file = "/workspace/mmcv/mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh"
    
    with open(cuda_file, 'r') as f:
        lines = f.readlines()
    
    # 最近邻插值函数（将成为第22-48行）
    nearest_code = """template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c) {
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[CUDA] Nearest sampling: h=%.2f w=%.2f\\n", (float)h, (float)w);
  }
  
  const int h_nearest = __float2int_rn(h);
  const int w_nearest = __float2int_rn(w);

  if (h_nearest < 0 || h_nearest >= height || 
      w_nearest < 0 || w_nearest >= width) {
    return 0;
  }

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int ptr = h_nearest * h_stride + w_nearest * w_stride + m * channels + c;
  
  return bottom_data[ptr];
}

"""
    
    # 构建新文件：前21行 + 新函数 + 第68行开始
    new_lines = lines[:21] + [nearest_code] + lines[67:]
    
    with open(cuda_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ 已替换第22-67行")
    print(f"✓ 新文件总行数: {len(new_lines)}")
    
    # 验证
    with open(cuda_file, 'r') as f:
        content = f.read()
    
    assert '#ifndef' in content and '#endif' in content
    assert 'Nearest sampling' in content
    print("✓ 验证通过")

if __name__ == '__main__':
    final_replace()

