#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# We use text-embedding-3-large & top-10 as the main RAG evaluation setting

models=(
    # 'facebook/contriever-msmarco'
    # 'bm25'
    'text-embedding-3-large'
)

data_files=(
    'data/testmini.json'
    'data/test.json'
)

top_ks=(
  # 3
  # 5
  10
)

for data_file in "${data_files[@]}"; do
  for model in "${models[@]}"; do
    for top_k in "${top_ks[@]}"; do
      echo "Running embedding for model: $model on $data_file set with top_k: $top_k"

      python retriever/retriever.py \
          --model_name "$model" \
          --data_file "$data_file" \
          --top_k "$top_k"
    done
  done
done
