#!/bin/bash

prompt_types=(
    "cot"
    "do"
)

settings=(
    "rag"
    "long_context"
)

subset='testmini'

for setting in "${settings[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        raw_dir="outputs/${subset}_outputs/${setting}/raw_${prompt_type}_outputs"
        processed_dir="outputs/${subset}_outputs/${setting}/processed_${prompt_type}_outputs"
        result_file="outputs/${subset}_outputs/results/${setting}_${prompt_type}_results.json"

        # remove result file if it exists
        if [ -f "$result_file" ]; then
            rm "$result_file"
        fi

        # Iterate over each file in the raw output directory
        for raw_file in "$raw_dir"/*; do
            filename=$(basename "$raw_file")
            
            python evaluation.py \
                --prediction_path "$raw_file" \
                --evaluation_output_dir "$processed_dir" \
                --prompt_type "$prompt_type" \
                --result_file "$result_file"

            echo "Finished evaluating $filename"
        done
    echo "Finished evaluating $prompt_type on $subset set"
    done
done