#!/usr/bin/env bash

# --- 固定的参数 ---
# 您可以在这里修改默认值
CONFIG="projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py"
CHECKPOINT="ckpts/bevformerV2-t1-24.pth"
GPUS="1"
QUANT_PARAMS="quantization_params.pth" # 假设量化参数文件在 BEVFormer 根目录
PORT=${PORT:-29504}

echo "--- 使用固定参数运行量化测试 ---"
echo "配置文件: $CONFIG"
echo "模型文件: $CHECKPOINT"
echo "量化参数: $QUANT_PARAMS"
echo "----------------------------------"

# --- 设置 PYTHONPATH ---
# 优先使用我们自己编译的 mmcv 源码，并包含 BEVFormer 项目路径
export PYTHONPATH=/workspace/mmcv_source:"$(dirname $0)/..":$PYTHONPATH

# --- 启动测试 ---
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_with_quantization.py $CONFIG $CHECKPOINT --launcher pytorch \
    --quantization-params $QUANT_PARAMS --eval bbox
