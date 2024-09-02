#!/bin/bash
raw_dir="outputs/testmini_outputs/rag_analysis/raw_outputs"
processed_dir="outputs/testmini_outputs/rag_analysis/processed_outputs"
result_file="outputs/testmini_outputs/rag_analysis/results.json"

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
        --prompt_type cot \
        --result_file "$result_file"

    echo "Finished evaluating $filename"
done


raw_dir="outputs/testmini_outputs/oracle/raw_cot_outputs"
processed_dir="outputs/testmini_outputs/oracle/processed_cot_outputs"
result_file="outputs/testmini_outputs/oracle/results.json"

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
        --prompt_type cot \
        --result_file "$result_file"

    echo "Finished evaluating $filename"
done
