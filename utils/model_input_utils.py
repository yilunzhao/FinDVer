from tqdm import tqdm
import tiktoken
import re
import os, json

def count_tokens(input_str):
    enc = tiktoken.encoding_for_model("gpt-4")
    input_token_len = len(enc.encode(input_str))
    return input_token_len

def prepare_input_document(reports_dir, report):
    report_path = os.path.join(reports_dir, report)
    report_data = json.load(open(report_path, 'r'))
    
    input_text = ""
    for i, paragraph in enumerate(report_data['context']):
        input_text += f"[paragraph id = {i}] {paragraph['context']}\n"
    return input_text

def get_context(example, reports_dir, retrieve_data):
    report_path = os.path.join(reports_dir, example['report'])
    report_data = json.load(open(report_path, 'r'))
    paragraphs = report_data['context']

    for retrieved_example in retrieve_data:
        if retrieved_example['example_id'] == example['example_id']:
            retrieval_evidence = [f"[paragraph id = {i}] {paragraphs[i]['context']}" for i in retrieved_example['retrieved_context']]
            break

    context = "\n".join(retrieval_evidence)
    return context


def prepare_do_model_input(example, reports_dir, retrieve_data):
    system_input = """
As a financial expert, your task is to assess the truthfulness of the given statement by determining whether it is entailed or refuted based on the provided financial document. You should directly output the entailment label ("entailed" or "refuted") without any intermediate steps.
"""

    if retrieve_data is not None:
        input_context = get_context(example, reports_dir, retrieve_data)
    else:
        input_context = prepare_input_document(reports_dir, example['report'])
    
    user_input = 'Financial Document:\n' + input_context + "\n\n"
    user_input += f"Statement to verify: {example['statement']}\n\nDirectly output the entailment label ('entailed' or 'refuted') based on the given statement and the document."
    return system_input, user_input

def prepare_cot_model_input(example, reports_dir, retrieve_data):
    system_input = """
As a financial expert, your task is to assess the truthfulness of the given statement by determining whether it is entailed or refuted based on the provided financial document. Follow these steps:
1. Carefully read the given context and the statement. 
2. Analyze the document, focusing on the relevant financial data or facts that related to the statement.
3. Document each step of your reasoning process to ensure your assessment is clear and thorough.
4. Conclude your analysis with a final determination. In your last sentence, clearly state your conclusion in the following format: "Therefore, the statement is {entailment_label}." Replace {entailment_label} with either 'entailed' (if the statement is supported by the document) or 'refuted' (if the statement contradicts the document).
"""
    user_input = 'Financial Document:\n'

    if retrieve_data is not None:
        input_context = get_context(example, reports_dir,retrieve_data)
    else:
        input_context = prepare_input_document(reports_dir, example['report'])
        

    user_input += input_context + "\n\n"
    user_input += f"Statement to verify: {example['statement']}\n\nLet's think step by step to verify the given statement."
    
    return system_input, user_input
    
def prepare_model_inputs(qa_data, prompt_type, model_name, api_based, reports_dir, tokenizer=None, retrieval_data=None):
    model_inputs = []
    for index, example in tqdm(enumerate(qa_data)):
        if prompt_type == "do":
            system_input, user_input = prepare_do_model_input(example, reports_dir, retrieval_data)
        else:
            system_input, user_input = prepare_cot_model_input(example, reports_dir, retrieval_data)
            
        models_without_system = ("gemma", "OLMo", "Mistral", "Mixtral", "starcoder2")
        if any(model in model_name for model in models_without_system):
            model_input = [
                {"role": "user", "content": system_input + "\n" + user_input}
            ]
        else:
            model_input = [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input}
            ]
        if not api_based:
            model_input = tokenizer.apply_chat_template(model_input, tokenize=False)
        
        model_inputs.append(model_input)
    
    system_input = system_input
    user_input = user_input
    # write to file
    # with open(f"{prompt_type}_example_input.txt", "w", encoding="utf-8") as f:
    #     f.write(f"System Input: {system_input}\n")
    #     f.write(f"User Input: {user_input}\n")

    return model_inputs