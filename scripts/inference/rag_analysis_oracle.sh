#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)

subset='testmini'

echo "Running oracle setting for model: GPT-4o on $subset set"
python run_llm.py \
    --model_name gpt-4o \
    --api \
    --prompt_type cot \
    --data_file "data/${subset}.json" \
    --output_dir "outputs/${subset}_outputs/oracle" \
    --oracle 

    
models=(
    'Qwen/Qwen2.5-72B-Instruct-AWQ'
    'TechxGenus/Mistral-Large-Instruct-2407-AWQ'
    'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'
)

for model in "${models[@]}"; do
    echo "Running oracle setting for model: $model on $subset set"
    python run_llm.py \
        --model_name "$model" \
        --gpu_memory_utilization 0.95 \
        --prompt_type cot \
        --data_file "data/${subset}.json" \
        --output_dir "outputs/${subset}_outputs/oracle" \
        --oracle 
done