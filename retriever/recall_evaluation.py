import os, json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--ground_truth_file", type=str, required=True)
    args = parser.parse_args()

    data = json.load(open(args.input_file))
    ground_truth_data = json.load(open(args.ground_truth_file))

    gt_dict = {}
    for example in ground_truth_data:
        example_id = example["example_id"]
        gt_dict[example_id] = set(example["relevant_context"])

    recalls = []
    for example in data:
        example_id = example["example_id"]
        retrieved_ids = set(example["retrieved_context"])

        matched = gt_dict[example_id].intersection(retrieved_ids)
        recalls.append(len(matched) / len(gt_dict[example_id]))

    recall = round(sum(recalls) / len(recalls) * 100, 2)


    file_name = os.path.basename(args.input_file)
    top_k = os.path.basename(os.path.dirname(args.input_file))
    output_dir = os.path.dirname(os.path.dirname(args.input_file))
    output_file = os.path.join(output_dir, f"results_{top_k}.json")
    
    if os.path.exists(output_file):
        results = json.load(open(output_file))
    else:
        results = {}

    results[file_name] = recall
    json.dump(results, open(output_file, "w"), indent=4)

    print(f"For {args.input_file}, recall is {recall}")
        
        
