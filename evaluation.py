import json
import os
import argparse
from tqdm import tqdm
from utils.evaluation_utils import *
import signal
from dotenv import load_dotenv
load_dotenv()

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Code execution took too long!")
signal.signal(signal.SIGALRM, timeout_handler)

def get_result(processed_data):
    # Initialize counters for each subset and each split
    results = {
        # "test": {"correct": 0, "correct_ie": 0, "correct_knowledge": 0, "correct_numeric": 0, "subset_counts": {"ie": 0, "knowledge": 0, "numeric": 0}},
        "testmini": {"correct": 0, "correct_ie": 0, "correct_knowledge": 0, "correct_numeric": 0, "subset_counts": {"ie": 0, "knowledge": 0, "numeric": 0}}
    }

    # Process each example in the data
    for example in processed_data:
        split = example["split"]
        subset = example["subset"]
        result = example["result"]

        # Update correct counts based on split and subset
        results[split]["correct"] += result
        if subset == "knowledge":
            results[split]["correct_knowledge"] += result
            results[split]["subset_counts"]["knowledge"] += 1
        elif subset == "ie":
            results[split]["correct_ie"] += result
            results[split]["subset_counts"]["ie"] += 1
        elif subset == "numeric":
            results[split]["correct_numeric"] += result
            results[split]["subset_counts"]["numeric"] += 1

    # Calculate average accuracies for each split and subset
    def calculate_accuracies(split_data, total_count):
        avg_acc = round((split_data["correct"] / total_count) * 100, 1) if total_count > 0 else 0
        avg_acc_ie = round((split_data["correct_ie"] / split_data["subset_counts"]["ie"]) * 100, 1) if split_data["subset_counts"]["ie"] > 0 else 0
        avg_acc_knowledge = round((split_data["correct_knowledge"] / split_data["subset_counts"]["knowledge"]) * 100, 1) if split_data["subset_counts"]["knowledge"] > 0 else 0
        avg_acc_numeric = round((split_data["correct_numeric"] / split_data["subset_counts"]["numeric"]) * 100, 1) if split_data["subset_counts"]["numeric"] > 0 else 0
        return avg_acc, avg_acc_ie, avg_acc_knowledge, avg_acc_numeric

    # Get total counts for test and validation splits
    # test_count = sum(results["test"]["subset_counts"].values())
    testmini_count = sum(results["testmini"]["subset_counts"].values())

    # Calculate accuracies for test and validation
    # test_accuracies = calculate_accuracies(results["test"], test_count)
    testmini_accuracies = calculate_accuracies(results["testmini"], testmini_count)

    return {
        # "test": test_accuracies[0],
        "testmini": testmini_accuracies[0],
        # "test_ie": test_accuracies[1],
        # "test_knowledge": test_accuracies[2],
        # "test_numeric": test_accuracies[3],
        "testmini_ie": testmini_accuracies[1],
        "testmini_knowledge": testmini_accuracies[2],
        "testmini_numeric": testmini_accuracies[3]
    }



def evaluate_cot_pred_file(prediction_data, client):
    output_data = []
    correct = 0
    prediction_data = extract_cot_answers(prediction_data, client)
    correct_ie, correct_knowledge, correct_numeric = 0, 0, 0
    for example in tqdm(prediction_data):
        pred = example["extracted_label"]
        gt = "entailed" if example["entailment_label"] else "refuted"
        example["result"] = (pred == gt)
        output_data.append(example)

    return output_data


def evaluate_do_pred_file(prediction_data, client, timeout_duration=3):
    output_data = []
    correct_ie, correct_knowledge, correct_numeric = 0, 0, 0
    correct = 0
    prediction_data = extract_do_answers(prediction_data, client)
    for example in tqdm(prediction_data):
        pred = example["extracted_label"]
        gt = "entailed" if example["entailment_label"] else "refuted"
        example["result"] = (pred == gt)
        output_data.append(example)
    
    return output_data
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str, required=True)
    parser.add_argument("--evaluation_output_dir", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["do", "cot"])
    parser.add_argument("--result_file", type=str, required=True)
    args = parser.parse_args()


    client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'),
    )


    file_name = os.path.basename(args.prediction_path)
    os.makedirs(args.evaluation_output_dir, exist_ok=True)
    output_path = os.path.join(args.evaluation_output_dir, file_name)
    

    if os.path.exists(output_path):
        processed_data = json.load(open(output_path, "r"))
        score = get_result(processed_data)
    else:
        prediction_data = json.load(open(args.prediction_path, "r"))
        
        eval_func = evaluate_cot_pred_file if args.prompt_type == "cot" else evaluate_do_pred_file
        outputs = eval_func(prediction_data, client)
        score = get_result(outputs)

        json.dump(outputs, open(output_path, "w"), indent=4, ensure_ascii=False)

        
    print(f"Accuracy: {score['testmini']}")

    model_name = file_name.split(".")[0]

    if os.path.exists(args.result_file):
        results = json.load(open(args.result_file))
    else:
        results = []

    for result in results:
        if result["model_name"] == model_name:
            results.remove(result)
            break

    entry = {"model_name": model_name}
    entry.update(score)

    results.append(entry)

    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    results = sorted(results, key=lambda x: x["testmini"], reverse=True)
    json.dump(results, open(args.result_file, "w"), indent=4, ensure_ascii=False)
