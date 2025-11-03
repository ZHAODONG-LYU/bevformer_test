# BEVFormer Profiling 脚本使用说明

脚本路径：`tools/profile_bevformer.py`

## 功能概览

该脚本用于分析 BEVFormer 推理过程中关键模块的耗时，重点包括：

- BEV Encoder 内部的 SCA / TCA / FFN 以及其它开销
- 所有 SCA 中的 MSDA 调用维度与耗时
- 整批 MSDA 的“数据搬运（IO）+ 核心计算（Kernel）”时间
- 可选生成饼状图与 Chrome Trace 以便可视化分析

## 环境与前置条件

1. **配置文件**：需使用带有 `model` 和 `data` 字段的完整配置，例如：
   `/home/lvzhaodong/bev_tensorRT/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py`
2. **权重文件（可选）**：如果仅关注耗时，可将 checkpoint 参数设为 `none` 跳过加载；若要加载模型参数，请确保 `.pth` 文件完整无损。
3. **数据路径**：配置中 `data_root` 及预处理文件需要正确存在。

## 基本用法

```bash
python tools/profile_bevformer.py \
  /home/lvzhaodong/bev_tensorRT/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py \
  none \
  --warmup-iters 1 \
  --iters 1 \
  --target-query 9662 \
  --output /workspace/BEVFormer/profiler_summary.json \
  --export-trace /workspace/BEVFormer/profiler_trace.json \
  --pie-output /workspace/BEVFormer/profiler_pie.png
```

主要参数说明：

| 参数 | 说明 |
| --- | --- |
| 第 1 个位置参数 | 配置文件路径 |
| 第 2 个位置参数 | checkpoint 路径；写 `none` 可跳过加载 |
| `--warmup-iters` | 热身轮数（默认 1） |
| `--iters` | 正式 profile 轮数（默认 1） |
| `--target-query` | 要捕获的 BEV query index，若无需指定可改成 `None` |
| `--output` | 汇总结果输出的 JSON 文件路径 |
| `--export-trace` | Chrome Trace 输出路径，便于在 `chrome://tracing` 或 Nsight 中查看 |
| `--pie-output` | 饼状图保存路径，用于展示 SCA/TCA/FFN/Other 占比 |
| `--msda-replay-iters` | MSDA 重放（IO + Kernel）时的重复次数（默认 10） |

## 输出结果说明

运行完成后，`profiler_summary.json` 中包含：

- `metrics.BEVEncoder`：BEV Encoder 总 CUDA 耗时
- `metrics.BEVEncoder_breakdown`：SCA / TCA / FFN / Other 的耗时与占比
- `metrics.MSDA`：真实推理过程中所有 MSDA 内核的总耗时
- `msda_calls`：本批次各层 SCA 的 MSDA 调用维度
- `msda_call_aggregate`：按层统计调用次数与维度
- `msda_full_call`：一次整批 MSDA 的 IO、Kernel、总耗时（默认捕获第 1 层或指定 query）
- `msda_full_call_per_layer`：每层 BEVLayer 的整批 MSDA IO、Kernel、总耗时
- `msda_full_call_totals`：将所有层的 IO、Kernel 时间求和后的总计
- `artifacts.pie_chart`：饼状图路径（若指定 `--pie-output`）

若指定 `--export-trace`，脚本还会生成 `profiler_trace.json`，可使用 Chrome 或 Nsight Systems 查看时间轴。

## 常见问题

- **配置报错（`Config must contain...`）**：说明配置为空或字段缺失，请确认使用了完整配置。
- **`EOFError: Ran out of input`**：checkpoint 文件为空或损坏，重新下载或改用 `none`。
- **未捕获 MSDA**：如果 `target_query` 在当前 batch 中不存在，`msda_full_call` 会提示未捕获，可将 `--target-query` 改为 `None` 或换一个存在的索引。
- **多轮平均**：可增大 `--iters` 与 `--msda-replay-iters`，获取更稳定的统计结果。

## 扩展使用

若希望分析更多模块，例如 backbone 或 decoder，可在对应模块前增加 `record_function("模块名")`，脚本会在 `metrics` 中自动新增统计条目。饼状图也可以根据需要扩展，展示更细粒度的耗时比例。

---

通过以上操作，可以快速拿到 BEV Encoder 中 SCA / TCA / FFN / Other 的耗时比例，以及 MSDA（含 IO 与 Kernel）在整条流水线中的占比，为后续的硬件加速或算法优化提供依据。

