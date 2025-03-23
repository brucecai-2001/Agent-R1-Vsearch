#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from collections import defaultdict
import os


def convert_ckpt_2_hf(base_path: str, step: str = "100", world_size: int = 2):
    """
    将FSDP格式的checkpoint转换为HuggingFace格式

    Args:
        base_path: 基础路径，包含所有checkpoint相关目录
        step: 训练步数
        world_size: 分布式训练的world_size
    """
    # 构建相关路径
    fsdp_checkpoint_path = os.path.join(base_path, f"global_step_{step}", "actor")
    huggingface_model_path = os.path.join(fsdp_checkpoint_path, "huggingface")
    output_path = os.path.join(
        base_path, "huggingface_checkpoint", f"checkpoint_global_step_{step}"
    )

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 加载并合并分布式checkpoint
    state_dict = defaultdict(list)
    for rank in range(world_size):
        filepath = os.path.join(
            fsdp_checkpoint_path, f"model_world_size_{world_size}_rank_{rank}.pt"
        )
        print(f"Loading checkpoint from: {filepath}")
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    # 合并张量
    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    # 加载模型配置和权重
    print(f"Loading config from: {huggingface_model_path}")
    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    # 保存模型和tokenizer
    print(f"Saving model to: {output_path}")
    model.save_pretrained(output_path, max_shard_size="10GB")

    print(f"Saving tokenizer to: {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)

    print("Conversion completed successfully!")
    return output_path


if __name__ == "__main__":
    base_path = "/data/wdy/Agent-R1/checkpoints/hotpotqa_qwen2.5-0.5b-instruct-bs128-mb32-gb4/ppo"
    converted_path = convert_ckpt_2_hf(base_path=base_path, step="100", world_size=2)
    print(f"Converted model saved to: {converted_path}")
