from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
import json
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
from transformers import AutoTokenizer
import random
from typing import Union
import asyncio
import openai
from utils.openai_utils import *
from utils.model_input_utils import prepare_model_inputs
from huggingface_hub import login
import dotenv
dotenv.load_dotenv()

login(token=os.getenv('HUB_TOKEN'))


def process_single_example_raw_outputs(outputs):
    processed_outputs = []
    assert len(outputs.outputs) == 1
    processed_outputs.append(outputs.outputs[0].text)
    return processed_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    
    # dataset and output
    parser.add_argument("--data_file", type=str, default="data/testmini.json")
    parser.add_argument("--output_dir", type=str, default="outputs")

    # retriever setting for simplong and complong
    parser.add_argument("--retriever", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--reports_dir", type=str, default="financial_reports")
    parser.add_argument("--retriever_output_dir", type=str, default="outputs/testmini_outputs/retriever_output")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--retriever_model_name", type=str, default="text-embedding-3-large", choices=["text-embedding-3-large", "contriever-msmarco", "bm25"])
    
    # llm setting
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=int, default=1.0)
    parser.add_argument("--prompt_type", type=str, default="cot", choices=["do", "cot"])
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_num_examples", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--quantization", type=str, default="")
    
    # api key
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--requests_per_minute", type=int, default=100)
    
    args = parser.parse_args()
    
    gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    data = json.load(open(args.data_file, "r"))
    
    if args.max_num_examples > 0:
        data = data.select(range(args.max_num_examples))
    
    assert not (args.retriever == True and args.oracle == True)

    if "testmini" in args.data_file:
        subset = "testmini"
    elif "test" in args.data_file:
        subset = "test"
    else:
        raise ValueError("Unknown dataset")
    
    suffix_model_name = args.model_name.split("/")[-1].replace(".", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.retriever and not args.oracle: # long-context setting
        output_dir = os.path.join(args.output_dir, f"raw_{args.prompt_type}_outputs")
        retrieval_data = None
        output_file = os.path.join(output_dir, f"{suffix_model_name}.json")
    elif args.oracle:
        output_dir = os.path.join(args.output_dir, f"raw_{args.prompt_type}_outputs")
        
        retrieval_data = json.load(open(args.data_file, "r"))
        for example in retrieval_data:
            example["retrieved_context"] = example["relevant_context"]
        output_file = os.path.join(output_dir, f"{suffix_model_name}.json")
    else:
        if "rag_analysis_output" in args.output_dir:
            output_file = os.path.join(args.output_dir, f"{suffix_model_name}-{args.retriever_model_name}-top_{args.topk}.json")
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(args.output_dir, f"{subset}_outputs", "rag", f"raw_{args.prompt_type}_outputs")
            os.makedirs(os.path.join(args.output_dir, f"{subset}_outputs"), exist_ok=True)
            
            output_file = os.path.join(output_dir, f"{suffix_model_name}.json")

        retrieved_filepath = os.path.join(args.retriever_output_dir, f"top_{args.topk}", f"{args.retriever_model_name}.json")
        if not os.path.exists(retrieved_filepath):
            raise FileNotFoundError(f"Retrieved file not found: {retrieved_filepath}")
        retrieval_data = json.load(open(retrieved_filepath, "r"))

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        exit()

    if not args.api:
        if args.quantization:
            llm = LLM(args.model_name,
                    tensor_parallel_size=gpu_count,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    trust_remote_code=True,
                    quantization=args.quantization)
        else:
            llm = LLM(args.model_name, 
                    tensor_parallel_size=gpu_count, 
                    dtype="half",
                    swap_space=16, 
                    gpu_memory_utilization=args.gpu_memory_utilization, 
                    trust_remote_code=True)
        
        sampling_params = SamplingParams(temperature = args.temperature, 
                                        top_p = args.top_p, 
                                        max_tokens = args.max_tokens)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, verbose=False, trust_remote_code=True)
        tokenizer.use_default_system_prompt = True
        model_inputs = prepare_model_inputs(data, args.prompt_type, args.model_name, args.api, args.reports_dir,tokenizer, retrieval_data) 
        
        outputs = llm.generate(model_inputs, sampling_params)
        raw_outputs = [process_single_example_raw_outputs(output) for output in outputs]
    
    else:
        # We use the One API proxy (https://github.com/songquanpeng/one-api) to centralize the proprietary API call.
        client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            # base_url="xxx",
        )

        model_inputs = prepare_model_inputs(data, args.prompt_type, args.model_name, args.api, args.reports_dir, None, retrieval_data)
        model_name = args.model_name              

        raw_outputs = asyncio.run(generate_from_openai_chat_completion( 
                                                    client = client,
                                                    messages = model_inputs,
                                                    engine_name = args.model_name, 
                                                    temperature = args.temperature, 
                                                    top_p = args.top_p, 
                                                    max_tokens = args.max_tokens,
                                                    requests_per_minute = args.requests_per_minute,))
    
    
    output_data = []
    for raw_output, qa in zip(raw_outputs, data):
        if type(raw_output) != list:
            qa["output"] = [raw_output]
        else:
            qa["output"] = raw_output
        output_data.append(qa)
        
    json.dump(output_data, open(output_file, "w"), indent=4, ensure_ascii=True)