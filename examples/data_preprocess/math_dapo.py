# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DAPO-Math-17k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math_dapo")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 加载 DAPO-Math-17k 数据集
    dapo_dataset = datasets.load_dataset(
        "/home/yanruiran/workspace/wdy/datasets/DAPO-Math-17k/"
    )

    # 加载 GSM8k 数据集作为验证集
    gsm8k_dataset = datasets.load_dataset("openai/gsm8k", "main")
    gsm8k_test = gsm8k_dataset["test"]

    # 获取 DAPO-Math-17k 的前5k条作为训练集
    train_dataset = dapo_dataset["train"].select(range(5000))

    # 获取 GSM8k 测试集的前100条作为验证集
    test_dataset = gsm8k_test.select(range(100))

    instruction_following = """You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""

    # 处理 DAPO-Math-17k 训练集
    def process_dapo(example, idx):
        # 从 prompt 中获取 content (question_raw)
        prompt_data = example.get("prompt", [])
        if prompt_data and len(prompt_data) > 0:
            message = prompt_data[0]
            question_raw = message.get("content", "")

            # 去掉最后的提示文本
            reminder_text = (
                '\n\nRemember to put your answer on its own line after "Answer:".'
            )
            if question_raw.endswith(reminder_text):
                question_raw = question_raw[: -len(reminder_text)]
            else:
                print(f"Warning: {question_raw} does not end with {reminder_text}")
        else:
            question_raw = ""

        question = question_raw + " " + instruction_following

        # 获取正确答案
        ground_truth = example.get("reward_model", {}).get("ground_truth", "")

        # 获取 data_source 和 ability
        data_source = example.get("data_source", "")
        ability = example.get("ability", "")

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": ability,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "question_raw": question_raw,
            },
        }
        return data

    # 从 GSM8k 测试集处理验证集
    def extract_solution(solution_str):
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        assert solution is not None
        final_solution = solution.group(0)
        final_solution = final_solution.split("#### ")[1].replace(",", "")
        return final_solution

    def process_gsm8k_test(example, idx):
        question_raw = example.pop("question")
        question = question_raw + " " + instruction_following

        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)

        data_source = "openai/gsm8k"

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "test",
                "index": idx,
                "answer": answer_raw,
                "question_raw": question_raw,
            },
        }
        return data

    processed_train = train_dataset.map(function=process_dapo, with_indices=True)
    processed_test = test_dataset.map(function=process_gsm8k_test, with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 确保本地目录存在
    os.makedirs(local_dir, exist_ok=True)

    processed_train.to_parquet(os.path.join(local_dir, "train.parquet"))
    processed_test.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
