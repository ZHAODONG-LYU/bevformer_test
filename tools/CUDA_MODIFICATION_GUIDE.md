# CUDA Kernel 修改指南：实现单点采样（最近邻插值）

## 目标
将 MSDA 的双线性插值改为最近邻插值（单点采样），以验证 FPGA 设计的计算量减少效果。

## 需要修改的文件

在你的 Docker 环境中，MMCV 的 CUDA kernel 文件位于：

```bash
# 1. 找到 MMCV 路径
MMCV_PATH=$(python3 -c "import mmcv; import os; print(os.path.dirname(mmcv.__file__))")

# 2. 需要修改的主要文件
$MMCV_PATH/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
```

---

## 修改步骤

### 步骤 1：备份原始文件

```bash
cd $MMCV_PATH/ops/csrc/common/cuda/
cp ms_deform_attn_cuda_kernel.cuh ms_deform_attn_cuda_kernel.cuh.backup
```

### 步骤 2：修改 CUDA kernel

在 `ms_deform_attn_cuda_kernel.cuh` 中找到 `ms_deform_attn_im2col_bilinear` 函数。

#### 原始代码（双线性插值）

```cuda
// 大约在第 50-150 行附近
template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data,
    const int &height,
    const int &width,
    const int &nheads,
    const int &channels,
    const scalar_t &h,
    const scalar_t &w,
    const int &m,
    const int &c) {
    
    // 计算 4 个邻近点的坐标
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    
    // 计算权重
    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh;
    const scalar_t hw = 1 - lw;
    
    // 边界检查
    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;
    
    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
    }
    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }
    
    // 双线性插值计算
    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    
    return val;
}
```

#### 修改后的代码（最近邻插值）

**方案 A：完全替换为最近邻（推荐用于测试）**

```cuda
template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data,
    const int &height,
    const int &width,
    const int &nheads,
    const int &channels,
    const scalar_t &h,
    const scalar_t &w,
    const int &m,
    const int &c) {
    
    // ========== 最近邻插值（单点采样）==========
    // Round 到最近的整数坐标
    const int h_nearest = int(h + 0.5);  // round
    const int w_nearest = int(w + 0.5);  // round
    
    // 边界检查
    if (h_nearest < 0 || h_nearest >= height || 
        w_nearest < 0 || w_nearest >= width) {
        return 0;  // 越界返回 0
    }
    
    // 计算偏移量
    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int base_ptr = m * channels + c;
    const int ptr = h_nearest * h_stride + w_nearest * w_stride + base_ptr;
    
    // 单点采样（只访问 1 次内存）
    return bottom_data[ptr];
}
```

**方案 B：添加编译开关（推荐用于生产）**

```cuda
// 在文件开头添加宏定义控制
#ifndef USE_NEAREST_SAMPLING
#define USE_NEAREST_SAMPLING 0  // 0=双线性插值, 1=最近邻插值
#endif

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data,
    const int &height,
    const int &width,
    const int &nheads,
    const int &channels,
    const scalar_t &h,
    const scalar_t &w,
    const int &m,
    const int &c) {
    
#if USE_NEAREST_SAMPLING
    // ========== 最近邻插值 ==========
    const int h_nearest = int(h + 0.5);
    const int w_nearest = int(w + 0.5);
    
    if (h_nearest < 0 || h_nearest >= height || 
        w_nearest < 0 || w_nearest >= width) {
        return 0;
    }
    
    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int base_ptr = m * channels + c;
    const int ptr = h_nearest * h_stride + w_nearest * w_stride + base_ptr;
    
    return bottom_data[ptr];
    
#else
    // ========== 原始双线性插值 ==========
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    
    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh;
    const scalar_t hw = 1 - lw;
    
    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;
    
    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
    }
    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }
    
    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    
    return val;
#endif
}
```

### 步骤 3：重新编译 MMCV

修改完成后，需要重新编译 MMCV 的 CUDA 扩展：

```bash
# 方法 1：清除缓存并重新导入（如果你之前用的这个方法）
cd $MMCV_PATH
rm -rf build/
rm -rf _ext*.so
python3 -c "import torch; import mmcv"

# 方法 2：强制重新编译 MMCV ops（推荐）
cd $MMCV_PATH/../  # 回到 site-packages 目录
python3 -m pip uninstall mmcv-full -y
python3 -m pip install mmcv-full==1.4.0 --no-cache-dir

# 方法 3：如果 MMCV 是从源码安装的
cd /path/to/mmcv/source/
python3 setup.py build_ext --inplace
python3 setup.py install
```

### 步骤 4：验证修改是否生效

```bash
# 运行测试，观察结果
cd /home/lvzhaodong/BEVFormer
./tools/dist_test.sh \
    projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
    ckpts/bevformerV2-t1-24.pth \
    1 --eval bbox
```

**预期结果**：
- 如果修改生效，mAP 应该降到 ~0.352（与之前 Python 层面的 round 结果一致）
- 如果修改未生效，mAP 仍然是 ~0.363（原始双线性插值）

---

## 验证计算量减少

修改生效后，你可以：

1. **运行 profile 脚本**，对比 MSDA kernel 时间：
   ```bash
   python tools/profile_bevformer.py \
       projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
       ckpts/bevformerV2-t1-24.pth \
       --iters 10 --warmup-iters 5 --target-query 9662
   ```

2. **预期性能提升**：
   - 内存访问：4次 → 1次（减少 75%）
   - 计算量：4次乘加 → 0次（减少 100%）
   - 总延迟：可能减少 20-40%（取决于内存带宽）

---

## 调试技巧

### 1. 添加 printf 调试

在 CUDA kernel 中添加 printf（只打印第一个 thread）：

```cuda
if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[CUDA Debug] Using NEAREST sampling: h=%.2f -> h_nearest=%d\n", h, h_nearest);
}
```

### 2. 检查编译选项

```bash
# 查看 MMCV 编译时的 CUDA flags
python3 -c "import mmcv; print(mmcv.__version__); print(mmcv.__file__)"
python3 -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### 3. 对比结果

| 方法 | mAP | 实现方式 | 计算量减少 |
|------|-----|---------|-----------|
| 原始双线性 | 0.363 | CUDA 4点插值 | - |
| Python Round + CUDA | 0.352 | Python round + CUDA 4点 | ❌ 无 |
| **CUDA 最近邻** | **0.352** | **CUDA 单点** | **✅ 75%** |

---

## 恢复原始代码

如果需要恢复：

```bash
cd $MMCV_PATH/ops/csrc/common/cuda/
cp ms_deform_attn_cuda_kernel.cuh.backup ms_deform_attn_cuda_kernel.cuh
# 重新编译（步骤 3）
```

---

## 注意事项

1. **备份原始文件**：修改前一定要备份！
2. **测试环境隔离**：建议在测试环境先验证
3. **版本记录**：记录修改的 MMCV 版本和 commit
4. **性能测试**：用 `profile_bevformer.py` 验证性能提升
5. **精度测试**：用 `dist_test.sh` 验证精度损失

---

## 参考

- MMCV 源码：https://github.com/open-mmlab/mmcv
- Deformable DETR 论文：https://arxiv.org/abs/2010.04159
- CUDA 编程指南：https://docs.nvidia.com/cuda/

