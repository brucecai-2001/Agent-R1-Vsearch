#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# 设置环境变量
export MODEL_NAME="/data/wdy/Agent-R1/checkpoints/hotpotqa_qwen2.5-0.5b-instruct-bs128-mb32-gb4/ppo/global_step_100_tmp/actor"
export TENSOR_PARALLEL_SIZE=1
export PORT=8020

# 启动 vLLM OpenAI 兼容服务器
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port $PORT \
    --dtype bfloat16 \
    --max-model-len 16384