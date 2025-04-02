export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2

python3 -m agent_r1.src.hotpotqa_infer \
    +model.checkpoint_dir=/data/wdy/Agent-R1/checkpoints/hotpotqa_qwen2.5-0.5b-instruct-bs128-mb16-gb4/ppo-new-reward \
    +model.step=60 \
    +model.world_size=2 \
    +model.tensor_parallel_size=1 \
    +model.gpu_memory_utilization=0.6 \
    +generation.temperature=0.8 \
    +generation.top_p=0.7 \
    +generation.top_k=50 \
    +generation.max_tokens=1024 \
    data.val_files=/data/wdy/Agent-R1/data/hotpotqa/validation.parquet \
    +infer.max_rows=3 \
    +infer.embedding_model=/data/wdy/Downloads/models/BAAI/bge-large-en-v1.5 \
    +infer.index=/data/wdy/Agent-R1/data/corpus/hotpotqa/index.bin \
    +infer.corpus=/data/wdy/Agent-R1/data/corpus/hotpotqa/hpqa_corpus.jsonl \
    +infer.output_file=/data/wdy/Agent-R1/results/hotpotqa_qwen2.5-0.5b-instruct-bs128-mb16-gb4-ppo-new-reward.json 