#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)


models=(
  'gpt-4o'
)

subset='testmini'

for model in "${models[@]}"; do
  echo "Running inference for model: $model on $set"
  requests_per_minute=100

  python run_llm.py \
      --model_name "$model" \
      --api \
      --prompt_type cot \
      --data_file "data/${subset}.json" \
      --output_dir "outputs/${subset}_outputs/long_context" \
      --requests_per_minute "$requests_per_minute"
    
  python run_llm.py \
      --model_name "$model" \
      --gpu_memory_utilization 0.9 \
      --api \
      --prompt_type do \
      --data_file "data/${subset}.json" \
      --output_dir "outputs/${subset}_outputs/long_context" \
      --requests_per_minute "$requests_per_minute"
done