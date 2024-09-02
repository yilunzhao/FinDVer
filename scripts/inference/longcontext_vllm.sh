#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)


models=(
  # Llama
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"

  # Qwen
  'Qwen/Qwen2-7B-Instruct'
  'Qwen/Qwen2.5-7B-Instruct'

  # Mistral
  'mistralai/Ministral-8B-Instruct-2410'
)

subset='testmini'

for model in "${models[@]}"; do
  echo "Running inference for model: $model on $set"

  python run_llm.py \
      --model_name "$model" \
      --gpu_memory_utilization 0.9 \
      --prompt_type cot \
      --data_file "data/${subset}.json" \
      --output_dir "outputs/${subset}_outputs/long_context"
  
  python run_llm.py \
      --model_name "$model" \
      --gpu_memory_utilization 0.9 \
      --prompt_type do \
      --data_file "data/${subset}.json" \
      --output_dir "outputs/${subset}_outputs/long_context"
done