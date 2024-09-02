#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)


models=(
  'gemini-1.5-pro'
  'claude-3-5-sonnet-20241022'
  'gpt-4o'
)

subsets=(
  'testmini'
  'test'
)

for model in "${models[@]}"; do
  for subset in "${subsets[@]}"; do
    echo "Running inference for model: $model, subset: $subset"
    requests_per_minute=100

    python run_llm.py \
        --model_name "$model" \
        --data_file "data/${subset}.json" \
        --retriever_output_dir "archieve_outputs/${subset}_outputs/retriever_output" \
        --gpu_memory_utilization 0.9 \
        --retriever \
        --api \
        --prompt_type cot \
        --output_dir outputs \
        --requests_per_minute "$requests_per_minute"
      
    python run_llm.py \
        --model_name "$model" \
        --data_file "data/${subset}.json" \
        --retriever_output_dir "archieve_outputs/${subset}_outputs/retriever_output" \
        --gpu_memory_utilization 0.9 \
        --retriever \
        --api \
        --prompt_type do \
        --output_dir outputs \
        --requests_per_minute "$requests_per_minute"
  done
done