import pandas as pd
import json
import numpy as np
import faiss
from FlagEmbedding import FlagAutoModel
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import os
import torch
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from json_repair import repair_json


def load_data(parquet_file: str, max_rows: Optional[int]) -> pd.DataFrame:
    """
    加载 Parquet 文件并截取指定行数。

    Args:
        parquet_file: Parquet 文件路径
        max_rows: 最大行数

    Returns:
        加载后的 DataFrame
    """
    print("正在读取 Parquet 文件...")
    df = pd.read_parquet(parquet_file)
    if "prompt" not in df.columns:
        print("字段 'prompt' 不存在")
        exit(1)
    if max_rows is not None:
        print(f"截取前 {max_rows} 行数据...")
        df = df.iloc[:max_rows]
    return df


def search_tool(
    query: str, embedding_model: FlagAutoModel, index: faiss.Index, corpus: List[str]
) -> str:
    """
    执行搜索工具，返回搜索结果。

    Args:
        query: 搜索查询字符串

    Returns:
        搜索结果的 JSON 字符串
    """

    def _format_results(results: List) -> str:
        """
        格式化搜索结果以便更好地阅读。

        Args:
            results: 搜索结果列表

        Returns:
            格式化后的搜索结果字符串
        """
        results_list = []
        for i, result in enumerate(results):
            results_list.append(corpus[result])
        return json.dumps({"results": results_list})

    embeddings = embedding_model.encode_queries([query])
    dist, ids = index.search(embeddings, 5)  # ids: b*5
    results_str = _format_results(ids[0])
    return results_str


def extract_answer(response_content: str) -> str:
    """
    从响应内容中提取 <answer> 标签之间的内容。

    Args:
        response_content: 响应内容字符串

    Returns:
        提取的答案字符串
    """
    if "<answer>" in response_content and "</answer>" in response_content:
        start_idx = response_content.find("<answer>") + len("<answer>")
        end_idx = response_content.find("</answer>")
        return response_content[start_idx:end_idx].strip()
    return "I don't know"


def extract_tool_call_query(response_content: str):
    """
    从响应内容中提取 <tool_call> 标签之间的 query 参数。

    Args:
        response_content: 响应内容字符串

    Returns:
        提取的 query 参数字符串，如果不存在则返回 None
        提取的 response，去除 </tool_call> 后续的无意义内容
    """
    if "<tool_call>" in response_content and "</tool_call>" in response_content:
        start_parsed_idx = response_content.find("<tool_call>")
        start_idx = start_parsed_idx + len("<tool_call>")
        end_idx = response_content.find("</tool_call>")
        end_parsed_idx = end_idx + len("</tool_call>")
        tool_call_json = response_content[start_idx:end_idx].strip()
        parsed_response_content = response_content[start_parsed_idx:end_parsed_idx].strip()
        try:
            tool_call_json_repair = repair_json(tool_call_json)
            tool_call_data = json.loads(tool_call_json_repair)
            if tool_call_data.get("name") == "search":
                return tool_call_data.get("arguments", {}).get("query"), parsed_response_content
        except json.JSONDecodeError as e:
            print(f"tool_call JSON 解析失败: {e}")
    return None, None


def format_messages(prompt_data: Any) -> List[Dict[str, str]]:
    """
    格式化提示数据为 OpenAI 消息格式。

    Args:
        prompt_data: 提示数据

    Returns:
        格式化后的消息列表
    """
    import numpy as np
    if isinstance(prompt_data, list):
        return prompt_data
    if isinstance(prompt_data, np.ndarray):
        return prompt_data.tolist()
    elif isinstance(prompt_data, dict):
        return [prompt_data]
    else:
        return [{"role": "user", "content": str(prompt_data)}]


def generate_response(
    tokenizer: AutoTokenizer,
    llm: LLM,
    messages: List[Dict[str, str]],
    sampling_params: SamplingParams,
) -> str:
    """使用vllm生成响应"""
    # 将消息列表转换为提示文本
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 使用vllm生成
    outputs = llm.generate(prompt, sampling_params)
    response = outputs[0].outputs[0].text
    return response


def run_inference(
    row: pd.Series,
    idx: int,
    tokenizer: AutoTokenizer,
    llm: LLM,
    sampling_params: SamplingParams,
    embedding_model: FlagAutoModel,
    index: faiss.Index,
    corpus: List[str],
) -> Dict[str, Any]:
    """单个样本的推理过程"""
    prompt_data = row["prompt"]
    messages = format_messages(prompt_data)
    messages = (
        [
            {
                "role": "system",
                "content": """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "search", "description": "Search for information on the internet using Wikipedia as a knowledge source.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

For each tool response, return the response within <tool_response></tool_response> XML tags:
<tool_response>
{tool_response}
</tool_response>
""",
            },
        ]
        + messages
    )

    max_rounds = 5
    current_round = 0
    history = []

    while current_round < max_rounds:
        try:
            round_info = {
                "round": current_round + 1,
                "messages": messages.copy(),
                "initial_response": None,
                "parsed_response": None,
                "tool_call_query": None,
                "tool_response": None,
                "final_answer": None,
            }

            response_content = generate_response(
                tokenizer, llm, messages, sampling_params
            )
            round_info["initial_response"] = response_content

            # 检查是否得到最终答案
            final_answer = extract_answer(response_content)
            if final_answer != "I don't know":
                round_info["parsed_response"] = response_content
                round_info["final_answer"] = final_answer  # 记录最终答案
                history.append(round_info)  # 将当前轮次信息加入历史
                print(f"第 {idx} 行推理完成。")
                return {
                    "index": idx,
                    "prompt": prompt_data,
                    "initial_response": response_content,
                    "parsed_response": response_content,
                    "final_answer": final_answer,
                    "tool_call_query": None,
                    "history": history,  # 返回完整的对话历史
                }

            # 检查是否需要调用工具
            tool_call_query, parsed_response_content = extract_tool_call_query(response_content)
            if tool_call_query:
                round_info["parsed_response"] = parsed_response_content
                round_info["tool_call_query"] = tool_call_query  # 记录工具调用查询
                print(f"正在为第 {idx} 行执行搜索工具...")
                tool_response = search_tool(
                    tool_call_query, embedding_model, index, corpus
                )
                print(f"第 {idx} 行搜索工具执行完成。")
                round_info["tool_response"] = tool_response  # 记录工具响应

                # 将工具响应拼入下一轮对话
                messages.append({"role": "assistant", "content": parsed_response_content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_response>\n{tool_response}\n</tool_response>",
                    }
                )

            history.append(round_info)  # 将当前轮次信息加入历史
            current_round += 1
        except Exception as e:
            print(f"第 {idx} 行推理失败: {e}")
            return {
                "index": idx,
                "prompt": prompt_data,
                "initial_response": f"ERROR: {str(e)}",
                "parsed_response": f"ERROR: {str(e)}",
                "final_answer": f"ERROR: {str(e)}",
                "tool_call_query": None,
                "history": history,  # 返回失败前的对话历史
            }

    # 如果超过最大轮数仍未得到最终答案
    return {
        "index": idx,
        "prompt": prompt_data,
        "initial_response": "对话轮数超过限制，未得到最终答案。",
        "parsed_response": "对话轮数超过限制，未得到最终答案。",
        "final_answer": "I don't know",
        "tool_call_query": None,
        "history": history,  # 返回完整的对话历史
    }


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

    # 如果 model.safetensors 存在，则直接返回 output_path
    if os.path.exists(os.path.join(output_path, "model.safetensors")):
        print(f"Model already converted to HuggingFace format at: {output_path}")
        return output_path

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


def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    将推理结果保存到 JSON 文件。

    Args:
        results: 推理结果列表
        output_file: 输出文件路径
    """

    def custom_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        else:
            return str(obj)

    print("Saving inference results...")
    
    # 确保文件所在目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=custom_serializer)
    print(f"All inference results have been saved to {output_file}")


@hydra.main(config_path="config", config_name="agent_trainer", version_base=None)
def main(config: DictConfig):
    """主入口函数"""
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # 转换checkpoint为HF格式
    base_path = config.model.checkpoint_dir
    step = config.model.step
    world_size = config.model.world_size
    hf_model_path = convert_ckpt_2_hf(base_path, step, world_size)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

    # 初始化vllm
    sampling_params = SamplingParams(
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        max_tokens=config.generation.max_tokens,
    )

    llm = LLM(
        model=hf_model_path,
        trust_remote_code=True,
        tensor_parallel_size=config.model.tensor_parallel_size,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
    )

    # 加载数据
    df = load_data(config.data.val_files, config.infer.max_rows)

    # 初始化搜索组件
    embedding_model = FlagAutoModel.from_finetuned(
        config.infer.embedding_model,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        devices="cpu",
    )
    index = faiss.read_index(config.infer.index)

    # 加载语料库
    corpus = []
    with open(config.infer.corpus, "r") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data["title"] + " " + data["text"])

    # 开始推理
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="推理进度"):
        result = run_inference(
            row, idx, tokenizer, llm, sampling_params, embedding_model, index, corpus
        )
        results.append(result)
        print(f"第 {idx} 行完成")

    # 保存结果
    save_results(results, config.infer.output_file)


if __name__ == "__main__":
    main()
