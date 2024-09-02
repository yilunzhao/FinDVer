#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)


models=(
  # Llama
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
  "meta-llama/Llama-3.2-3B-Instruct"

  # Phi-3
  "microsoft/Phi-3.5-mini-instruct"

  # Deepseek
  'deepseek-ai/DeepSeek-V2-Lite-Chat'

  # Qwen
  'Qwen/Qwen2.5-7B-Instruct'
  'Qwen/Qwen2.5-72B-Instruct'

  # Mistral
  'mistralai/Mistral-7B-Instruct-v0.3'
  'mistralai/mathstral-7B-v0.1'
  'mistralai/Mistral-Large-Instruct-2407'

  # chatglm
  'THUDM/glm-4-9b-chat'
)

subsets=(
  'testmini'
  'test'
)

for model in "${models[@]}"; do
  for subset in "${subsets[@]}"; do
    echo "Running inference for model: $model on $set"

    python run_llm.py \
        --model_name "$model" \
        --data_file "data/${subset}.json" \
        --retriever_output_dir "archieve_outputs/${subset}_outputs/retriever_output" \
        --gpu_memory_utilization 0.9 \
        --retriever \
        --prompt_type cot \
        --output_dir outputs
      
    python run_llm.py \
        --model_name "$model" \
        --data_file "data/${subset}.json" \
        --retriever_output_dir "archieve_outputs/${subset}_outputs/retriever_output" \
        --gpu_memory_utilization 0.9 \
        --retriever \
        --prompt_type do \
        --output_dir outputs
  done
done