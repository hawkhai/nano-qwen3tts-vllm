#!/bin/bash
# TTS 运行脚本 - 禁用 cuDNN SDPA backend

# 设置环境变量
export TORCH_SDPA_BACKEND="flash_attention,mem_efficient,math"
export TORCH_CUDNN_SDPA_ENABLED="0"

# 默认参数
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
TEXT="${TEXT:-人工智能技术的发展日新月异}"
SPEAKER="${SPEAKER:-Vivian}"
LANGUAGE="${LANGUAGE:-Chinese}"
GPU_MEM="${GPU_MEM:-0.45}"

# 运行
cd "$(dirname "$0")"
python examples/custom_voice_example.py \
    --model-path "$MODEL_PATH" \
    --text "$TEXT" \
    --speaker "$SPEAKER" \
    --language "$LANGUAGE" \
    --gpu-memory-utilization "$GPU_MEM" \
    "$@"
