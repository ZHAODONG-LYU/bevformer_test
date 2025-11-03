# BEVFormer MSDA Int8 量化工具

本工具用于对 BEVFormer 中的 MSDA (Multi-Scale Deformable Attention) 模块的 value feature map 进行 int8 量化，并评估量化对模型精度的影响。

## 背景

MSDA 是 BEVFormer 的核心创新模块，对其进行硬件加速（例如使用 int8 计算）可以显著提升推理速度。在设计硬件加速器之前，需要先验证 int8 量化对模型精度的影响。

本工具采用 **PTQ (Post-Training Quantization，训练后量化)** 方法，无需重新训练模型，只需：
1. 使用少量样本统计 value feature map 的数值范围
2. 计算量化参数（scale）
3. 在推理时模拟 int8 量化效果（量化 + 反量化）
4. 评估量化前后的精度差异

## 工具说明

### 1. `collect_quantization_calibration.py`
收集校准数据，统计 value feature map 的数值范围，计算量化 scale。

### 2. `test_with_quantization.py`
使用量化参数进行推理测试，模拟 int8 量化效果。

### 3. `dist_test_quantized.sh`
分布式测试脚本（支持多卡）。

---

## 使用流程

### Step 1: 收集校准数据

首先需要用一部分验证集样本（例如 100 张图）来统计 value feature map 的数值范围。

#### 单卡运行：

```bash
cd /home/lvzhaodong/BEVFormer

python tools/collect_quantization_calibration.py \
    projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
    ckpts/bevformerV2-t1-24.pth \
    --calib-samples 100 \
    --output quantization_params.pth \
    --method max \
    --seed 0
```

#### 参数说明：
- `--calib-samples`: 用于校准的样本数量（默认 100，建议 100-500）
- `--output`: 输出的量化参数文件路径
- `--method`: 计算 scale 的方法
  - `max`: 使用最大值（简单、保守）
  - `percentile`: 使用分位数（更鲁棒，避免 outlier 影响）
- `--percentile`: 使用 `percentile` 方法时的分位数（默认 99.9）
- `--seed`: 随机种子

#### 输出示例：

```
[Calibration] 开始收集校准数据，使用 100 个样本...
[Calibration] 已处理 10/100 个样本
[Calibration] 已处理 20/100 个样本
...
[Calibration] 计算量化 scale（方法：max）...
[Calibration] 量化参数已保存到: quantization_params.pth
[Calibration] Scale shape: torch.Size([8, 4])
[Calibration] Scale 统计:
  - Min: 0.012345
  - Max: 0.234567
  - Mean: 0.098765

详细 scale 值 (head × level):
tensor([[0.1234, 0.1456, 0.1678, 0.1890],
        [0.1123, 0.1345, 0.1567, 0.1789],
        ...])
```

生成的 `quantization_params.pth` 包含：
- `scales`: [num_heads, num_levels] 的 scale 张量
- `num_heads`: head 数量（8）
- `num_levels`: level 数量（4）
- `method`: 使用的方法
- `calib_samples`: 校准样本数
- 其他元信息

---

### Step 2: 使用量化参数进行测试

#### 单卡测试：

```bash
cd /home/lvzhaodong/BEVFormer

python tools/test_with_quantization.py \
    projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
    ckpts/bevformerV2-t1-24.pth \
    --quantization-params quantization_params.pth \
    --eval bbox
```

#### 多卡测试（使用你熟悉的方式）：

```bash
cd /home/lvzhaodong/BEVFormer

./tools/dist_test_quantized.sh \
    projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
    ckpts/bevformerV2-t1-24.pth \
    1 \
    --quantization-params quantization_params.pth \
    --eval bbox
```

#### 参数说明：
- `--quantization-params`: 量化参数文件（由 Step 1 生成）
- `--eval`: 评估指标（例如 `bbox`）
- `--out`: 可选，保存结果到文件
- 其他参数与原来的 `test.py` 相同

#### 输出示例：

```
[Quantization] 加载量化参数: quantization_params.pth
[Quantization] Scale shape: torch.Size([8, 4])
[Quantization] 方法: max
[Quantization] 校准样本数: 100
[Quantization] Patched SpatialCrossAttention.forward
[Quantization] 开始测试（使用 int8 量化的 MSDA value）...
...
[Quantization] 量化操作执行次数: 234

==================================================
量化后的评估结果 (bbox):
==================================================
NDS: 0.5234
mAP: 0.4567
...
==================================================
```

---

### Step 3: 对比量化前后的精度

#### 量化前（原始模型）：

```bash
./tools/dist_test.sh \
    projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
    ckpts/bevformerV2-t1-24.pth \
    1 --eval bbox
```

#### 量化后：

```bash
./tools/dist_test_quantized.sh \
    projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
    ckpts/bevformerV2-t1-24.pth \
    1 \
    --quantization-params quantization_params.pth \
    --eval bbox
```

对比两次的 `NDS` 和 `mAP`，评估 int8 量化的精度损失。

---

## 量化原理

### 量化公式

对于 value feature map `V`（float32），量化到 int8：

```
scale = max(|V|) / 127
V_quantized = round(V / scale).clamp(-128, 127)
V_dequantized = V_quantized * scale
```

- **量化粒度**：按 `[head, level]` 统计，不跨 channel（因为 sampling_locations 和 attention_weights 对所有 channel 是共享的）
- **量化范围**：int8 的对称量化，范围 [-128, 127]
- **反量化**：为了在现有 PyTorch 框架中运行，量化后立即反量化回 float32，但精度已降低到 int8 的 256 个离散级别

### 为什么要反量化？

- **目的**：验证精度损失，而不是真正加速
- **原因**：PyTorch 的 MSDA CUDA kernel 是 float32 实现，直接输入 int8 会自动转成 float32
- **硬件加速**：如果要真正加速，需要修改 CUDA kernel，使用 int8 × int8 的 GEMM（例如 Tensor Core），这超出了本工具的范围

---

## 常见问题

### Q1: 校准样本数量选多少合适？
**A**: 一般 100-500 个样本就足够了。更多样本会更准确，但收集时间也更长。

### Q2: `max` 和 `percentile` 方法哪个更好？
**A**: 
- `max`：简单、保守，不会出现 clipping（截断）
- `percentile`：更鲁棒，可以忽略少数 outlier，一般精度更高

建议先用 `max` 验证，如果精度损失较大，再尝试 `percentile=99.9`。

### Q3: 量化后精度掉多少算合理？
**A**: 
- **<1% NDS/mAP 下降**：非常好，可以直接硬件加速
- **1-2% 下降**：可以接受，视具体应用场景
- **>2% 下降**：可能需要更精细的量化策略（例如 per-channel、混合精度）

### Q4: 如何进一步降低精度损失？
**A**: 可以尝试：
1. 使用 `percentile` 方法（99.9 或 99.95）
2. 增加校准样本数量
3. 使用 QAT (Quantization-Aware Training)，在训练时模拟量化
4. 混合精度：只量化部分层，保留关键层为 float32

### Q5: 这个工具会加速推理吗？
**A**: **不会**。本工具只是模拟 int8 量化的精度损失，计算仍然是 float32。真正的加速需要修改 CUDA kernel，使用 int8 GEMM。

### Q6: 如何验证量化确实生效了？
**A**: 查看输出中的 `量化操作执行次数`，应该大于 0。你也可以在代码中打印量化前后的数值对比。

---

## 技术细节

### Scale 的粒度

本工具按 `[head, level]` 统计 scale，原因：
- **head**: 每个 attention head 学习不同的特征，数值范围可能不同
- **level**: 不同的特征层级（例如 1/8, 1/16, 1/32, 1/64 分辨率）数值范围也不同
- **不跨 channel**: 因为 MSDA 的 `sampling_locations` 和 `attention_weights` 对所有 channel 是共享的，如果每个 channel 用不同 scale 会破坏这个结构

### 为什么不量化权重？

- `value` 是动态计算的（`value = value_proj(input)`），每张图不同
- `value_proj` 的权重可以量化，但那是**权重量化**，不是 **activation 量化**
- 对于 MSDA 硬件加速，activation 的量化（即 value feature map）更关键，因为它是每次推理都要计算的

---

## 下一步：硬件加速

如果 int8 量化的精度可以接受，可以考虑：

1. **修改 MSDA CUDA kernel**：
   - 让 kernel 直接接受 int8 输入
   - 使用 int8 × int8 累加（Tensor Core 或自定义 kernel）
   - 输出时再反量化

2. **使用 TensorRT INT8**：
   - 导出 ONNX 模型
   - 使用 TensorRT 的 INT8 calibration
   - 部署到 NVIDIA GPU

3. **自定义硬件加速器（FPGA/ASIC）**：
   - 根据 MSDA 的计算特点设计专用电路
   - 使用 int8 运算单元
   - 优化数据搬运和访存

---

## 参考资料

- **BEVFormer 论文**: [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270)
- **Deformable DETR 论文**: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
- **量化综述**: [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)

---

## 联系与反馈

如有问题或建议，请查看代码注释或修改脚本以适应你的需求。


