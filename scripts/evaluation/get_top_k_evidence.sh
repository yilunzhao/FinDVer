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

subsets=(
  'testmini'
  'test'
)

for set in "${subsets[@]}"; do
  for model in "${models[@]}"; do
      for k in "${top_ks[@]}"; do
      echo "Running embedding for model: $model on $set set with top_k: $k"

      model_name=$(echo $model | awk -F'/' '{print $NF}')
      input_file="outputs/${set}_outputs/retriever_output/all/${model_name}.json"
      output_dir="outputs/${set}_outputs/retriever_output/top_$k"
      python retriever/get_top_n.py \
          --input_file "$input_file" \
          --output_dir "$output_dir" \
          --top_k "$k"
    done
  done
done