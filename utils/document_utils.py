import tiktoken
from glob import glob
import random
import json
import re
import signal

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Code execution took too long!")
signal.signal(signal.SIGALRM, timeout_handler)


def prepare_selected_paragraphs(report_filepath, paragraph_ids):
    data = json.load(open(report_filepath, 'r'))
    paragraphs = data['paragraphs']
    tables = data['tables']

    selected_paragraphs = []
    for idx in paragraph_ids:
        if '##table' in paragraphs[idx]:
            table_idx = int(re.findall(r'\d+', paragraphs[idx])[0])
            selected_paragraphs.append(f"[table id = {idx}] {tables[table_idx]}")
        else:
            selected_paragraphs.append(f"[paragraph id = {idx}] {paragraphs[idx]}")

    return selected_paragraphs

def prepare_selected_paragraphs_tables(report_filepath, paragraph_ids, table_ids):
    data = json.load(open(report_filepath, 'r'))
    paragraphs = data['paragraphs']
    tables = data['tables']

    selected_paragraphs = []
    for idx in paragraph_ids:
        selected_paragraphs.append(f"[paragraph id = {idx}] {paragraphs[idx]}")

    for idx in table_ids:
        selected_paragraphs.append(f"[table id = {idx}] {tables[idx]}")

    return selected_paragraphs


def prepare_input_document(paragraphs, tables, token_limit=8000):
    input_text = 'Report:\n'

    if random.random() < 0.5:
        index = random.randint(0, len(paragraphs) - 1)
        index_list = range(index, len(paragraphs))
    else:
        index_list = random.sample(range(0, len(paragraphs)), min(100, len(paragraphs)))
        index_list.sort()

    input_index = []
    for j in index_list:
        tmp = input_text
        if '##table' in paragraphs[j]:
            table_idx = int(re.findall(r'\d+', paragraphs[j])[0])
            tmp += f"[table id = {table_idx}] {tables[table_idx]}\n"
        else:
            if len(paragraphs[j].split()) < 20:
                continue
            tmp += f"[paragraph id = {j}] {paragraphs[j]}\n"
        input_index.append(j)
        tokens = calculate_token(tmp)
        if tokens < token_limit:
            input_text = tmp
            if j == len(paragraphs) - 1:
                break
        else:
            break

    return input_text

def within_token_limit(paragraphs, tables, token_limit=80000):
    context = ''
    for i in range(len(paragraphs)):
        if '##table' in paragraphs[i]:
            context += tables[int(re.findall(r'\d+', paragraphs[i])[0])] + '\n'
        else:
            context += paragraphs[i] + '\n'
    
    if count_tokens(context) > token_limit:
        return False
    else:
        return True


def execute_code_from_string(code_string):
    # Define a local environment for the execution of the code string
    local_env = {}
    try:
        # Execute the string as Python code
        signal.alarm(5)  # set the alarm
        exec(code_string, {}, local_env)
        # Return the result of the function 'calculate' if it exists
        if 'calculate' in local_env:
            value = local_env['calculate']()
            signal.alarm(0)
            if isinstance(value, int) or isinstance(value, float):
                return round(value, 2)
            else:
                return value
        else:
            return None
    except (TimeoutException, Exception, OverflowError) as e:
        signal.alarm(0)
        print(e)
        return None