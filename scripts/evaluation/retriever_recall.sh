#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

models=(
    'facebook/contriever-msmarco'
    'bm25'
    'text-embedding-3-large'
)

top_ks=(
  3
  5
  10
)


for model in "${models[@]}"; do
    for k in "${top_ks[@]}"; do
    echo "Running embedding for model: $model on $set set with top_k: $k"

    model_name=$(echo $model | awk -F'/' '{print $NF}')
    
    input_file="outputs/testmini_outputs/retrieved_output/top_${k}/${model_name}.json"
    python retriever/recall_evaluation.py \
        --input_file "$input_file" \
        --ground_truth_file data/testmini.json
  done
done