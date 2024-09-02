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

subset='testmini'

# GPT-4o
for retrieval_model in "${retrievers[@]}"; do
    for k in "${top_ks[@]}"; do
        echo "Running embedding for model: $model on $set set with top_k: $k"
        python run_llm.py \
            --model_name gpt-4o \
            --api \
            --prompt_type cot \
            --data_file "data/${subset}.json" \
            --output_dir "outputs/${subset}_outputs/rag_analysis_output/raw_outputs" \
            --retriever \
            --retriever_model_name "$retrieval_model" \
            --topk "$k" \
            --requests_per_minute 100
    done
done