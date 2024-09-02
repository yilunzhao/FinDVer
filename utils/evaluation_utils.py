import sys
import json
import re
import ast
from sympy import Rational
import numpy as np
import os
from time import sleep
from tqdm import tqdm
import math
import faulthandler
import openai
import asyncio
from utils.openai_utils import *
from tqdm import tqdm
import random
from pydantic import BaseModel

class EntailmentLabel(BaseModel):
    entailment_label: str

# helper function for evaluating CoT predictions
def extract_cot_answers(examples, client):
    instruction = """Extract the final entailment label from the model's response during claim verification. Return one of the following labels: 'entailed', 'refuted', or 'none'. Return 'none' only if the model does not provide a valid entailment.

    Organize your response in the following json format:
    {
        "entailment_label": str // 'entailed', 'refuted', or 'none'
    }
    """
    model_inputs = []
    for example in examples:
        user_input = f"Model Response:\n{example['output']}\n"
        chat = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": user_input},
        ]
        model_inputs.append(chat)
    
    outputs = asyncio.run(generate_from_openai_chat_completion_parse(
                                                    client = client, 
                                                    messages = model_inputs,
                                                    engine_name = "gpt-4o-mini", 
                                                    max_tokens = 16,
                                                    requests_per_minute = 5000,
                                                    response_format=EntailmentLabel))
                                                    
    
    for example, output in zip(examples, outputs):
        label = output.parsed.entailment_label
        if label in ["entailed", "refuted"]:
            example["extracted_label"] = label
        elif label == "none":
            example["extracted_label"] = random.choice(["entailed", "refuted"])
        else:
            print(f"Invalid entailment label: {label}")
            example["extracted_label"] = random.choice(["entailed", "refuted"])
        
    return examples

def extract_do_answers(examples, client):
    for example in examples:
        output = example["output"][0]
        if "entail" in output.lower():
            example["extracted_label"] = "entailed"
        elif "refut" in output.lower():
            example["extracted_label"] = "refuted"
        else:
            example["extracted_label"] = random.choice(["entailed", "refuted"])
        
    return examples