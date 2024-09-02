#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)


top_ks=(
  3
  5
  10
)
retrievers=(
    'bm25'
    'text-embedding-3-large'
    'contriever-msmarco'
)

models=(
    'Qwen/Qwen2.5-72B-Instruct-AWQ'
    'TechxGenus/Mistral-Large-Instruct-2407-AWQ'
    'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'
)

subset='testmini'

for model in "${models[@]}"; do
    for retrieval_model in "${retrievers[@]}"; do
        for k in "${top_ks[@]}"; do
            echo "Running embedding for model: $model on $set set with top_k: $k"
            python run_llm.py \
                --model_name "$model" \
                --gpu_memory_utilization 1 \
                --prompt_type cot \
                --data_file "data/${subset}.json" \
                --output_dir "outputs/${subset}_outputs/rag_analysis_output/raw_outputs" \
                --retriever \
                --retriever_model_name "$retrieval_model" \
                --topk "$k"
        done
    done
done