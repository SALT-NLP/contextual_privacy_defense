import os
import json
from glob import glob
from datasets import Dataset
import re
from modules.agent_instruct import get_instruct_agent_prompt, sanitize_memory
from modules.agent_guard import get_guard_agent_prompt
from modules.agent_guard import sanitize_memory as sanitize_guard_memory
from utils import process_str_to_json
from typing import List, Dict

def get_content(content_dict: Dict, raise_exception: bool = True) -> str:
    if "content" in content_dict:
        return content_dict["content"]
    elif "body" in content_dict:
        return content_dict["body"]
    elif "message" in content_dict:
        return content_dict["message"]
    else:
        if not raise_exception:
            return ""
        raise ValueError(f"Unknown content format in data.  {content_dict}")

def extract_agent_memory(log_file_path):
    """Extract the last Agent Memory from log file"""
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    memory_pattern = r'Agent Memory:\s*\n.*?\((\[.*?\]), (\d+)\)'
    matches = re.findall(memory_pattern, content, re.DOTALL)
    if not matches:
        return None, None
    last_memory_str, token_count = matches[-1]
    try:
        memory_data = eval(last_memory_str)
        return memory_data, int(token_count)
    except:
        return None, None

def prepare_dataset_instruct(example_folder: str, use_mixed_objective: bool) -> List[Dict]:
    files = glob(os.path.join(example_folder, "example_v*", "example_*", "log_*", "m_evaluation_0.json"))
    contexts = []
    for file in files:
        example_id = int(re.search(r"example_(\d+)", file).group(1))
        # if example_id not in [16, 44, 73, 91, 93]:
        #     continue
        eval_data = json.load(open(file, 'r', encoding='utf-8'))
        config = json.load(open(os.path.join(os.path.dirname(os.path.dirname(file)), "config.json"), 'r', encoding='utf-8'))
        agent_mem = extract_agent_memory(file.replace("m_evaluation_0.json", "agent1.log"))[0]
        eval_lst = eval_data['evaluations']
        idx = None
        length = 0
        flag = -1
        for i, eval_item in enumerate(eval_lst):
            length += 1
            if use_mixed_objective:
                if eval_item["shared_shareable_items"] and flag == -1:
                    flag = eval_item["index"]
                if eval_item["shared_non_shareable_items"]:
                    if flag != -1:
                        idx = eval_item["index"]
                    break
            elif eval_item["shared_non_shareable_items"]:
                idx = eval_item["index"]
                break
        if idx is None:
            continue
        assert eval_data["history"][idx - 1]["user_id"] == config["simulation_config"]["data_sender_agent"]["concrete_name"].lower().replace(' ', '_')
        content = get_content(eval_data["history"][idx - 1]["details"])
        last_instruction = 0

        for i, msg in enumerate(agent_mem):
            if msg["role"] == "user":
                last_instruction = i
            if "tool_calls" in msg and msg["tool_calls"]:
                msg_content = process_str_to_json(msg["tool_calls"][0]["function"]["arguments"])
                msg_content = get_content(msg_content, raise_exception=False)
                if content not in msg_content:
                    continue
                contexts.append({ \
                    "prompt": [            
                        {"role": "system", "content": "You are a helpful AI assistant good at analyzing privacy risks and generating instructions for other agents."},
                        {"role": "user", "content": get_instruct_agent_prompt(sanitize_memory(agent_mem[:last_instruction])) }
                    ],
                    "memory": agent_mem[:last_instruction],
                    "path": os.path.dirname(file),
                    "original_content": content, 
                    "eval_lst_len": length,
                    "_label": -1,
                })
                assert length > 0
                break
    contexts = [item for item in contexts if len(item["prompt"][1]["content"].split()) <= 3000]
    print(f"Prepared dataset instruct size: {len(contexts)}")
    return contexts

def prepare_dataset_guard(example_folder: str) -> List[Dict]:
    files = glob(os.path.join(example_folder, "example_v*", "example_*", "log_*", "m_evaluation_0.json"))
    contexts = []
    for file in files:
        example_id = int(re.search(r"example_(\d+)", file).group(1))
        if example_id not in [16, 44, 73, 91, 93]:
            continue

        eval_data = json.load(open(file, 'r', encoding='utf-8'))
        config = json.load(open(os.path.join(os.path.dirname(os.path.dirname(file)), "config.json"), 'r', encoding='utf-8'))
        agent_mem = extract_agent_memory(file.replace("m_evaluation_0.json", "agent1.log"))[0]
        eval_lst = eval_data['evaluations']
        idx = None
        label = 1
        for i, eval_item in enumerate(eval_lst):
            if eval_item["shared_non_shareable_items"]:
                idx = eval_item["index"]
                label = 1
                break
            if eval_item["shared_shareable_items"] and eval_item["shared_non_shareable_items"] == []:
                idx = eval_item["index"]
                label = 0
                break
        if idx is None:
            continue
        assert eval_data["history"][idx - 1]["user_id"] == config["simulation_config"]["data_sender_agent"]["concrete_name"].lower().replace(' ', '_')
        content = get_content(eval_data["history"][idx - 1]["details"])
        for i, msg in enumerate(agent_mem):
            if "tool_calls" in msg and msg["tool_calls"]:
                msg_content = process_str_to_json(msg["tool_calls"][0]["function"]["arguments"])
                func_name = msg["tool_calls"][0]["function"]["name"]
                func_args = msg_content.copy()
                msg_content = get_content(msg_content, raise_exception=False)
                if content not in msg_content:
                    continue
                latest_message = ""
                if func_name == "Messenger_send_message":
                    latest_message = f"Message to {func_args['recipient_id']}:\n- {func_args['message']}"
                elif func_name == "Gmail_send_email":
                    latest_message = f"Email to {func_args['to_address']}:\n- {func_args['subject']}: {func_args['body']}"
                elif func_name == "Facebook_create_post":
                    latest_message = f"Post created:\n- {func_args['content']}"
                temp = get_guard_agent_prompt(sanitize_guard_memory(agent_mem[:i+2]), config["simulation_config"]["data_sender_agent"]["concrete_name"], latest_message)
                contexts.append({ \
                    "prompt": [
                        {"role": "system", "content": "You are a helpful assistant that helps to monitor and analyze the data sharing activities between agents to prevent sensitive data leakage."},
                        {"role": "user", "content": temp}
                    ],
                    "memory": [
                        {"233": temp}
                    ],
                    "path": os.path.dirname(file),
                    "original_content": "EMPTY",
                    "eval_lst_len": -1,
                    "_label": label,
                })
                break
    dataset = contexts
    dataset = [item for item in dataset if len(item["prompt"][1]["content"].split(' ')) <= 3000]
    print(f"Prepared dataset guard size: {len(dataset)}")
    print(f"Label distribution: {[sum(1 for d in dataset if d['_label'] == lbl) for lbl in [0,1]]}")
    return dataset

def prepare_dataset(args, output_dir) -> Dataset:
    dataset = []
    if args.dataset_instruct_folder:
        dataset.extend(prepare_dataset_instruct(args.dataset_instruct_folder, args.use_mixed_objective))
    if args.dataset_guard_folder:
        assert not args.use_mixed_objective, "Mixed objective is only for instruct dataset."
        dataset.extend(prepare_dataset_guard(args.dataset_guard_folder))
    data_dict = {}
    data_dict["index"] = list(range(len(dataset)))
    data_dict["output_dir"] = [output_dir] * len(dataset)
    for key in dataset[0].keys():
        data_dict[key] = [d[key] for d in dataset]
    dataset = Dataset.from_dict(data_dict)
    return dataset

def debug_dataset(dataset):
    print("=== Dataset Debug ===")
    sample = dataset[2]
    print(f"Sample keys: {sample.keys()}")
    print(len(dataset))

    print(f"First prompt: {sample.get('prompt')[1]['content']}")
    print(f"First eval_lst_len: {sample.get('eval_lst_len')}")
    print(f"First original instruction: {sample.get('original_content')}")
    print("=" * 50)

if __name__ == "__main__":
    dataset = prepare_dataset_instruct(example_folder="logs-trained/dataset_instruct", use_mixed_objective=True)
    debug_dataset(dataset)
    # debug purpose
    # dataset = prepare_dataset_instruct(example_folder="logs-trained/dataset_instruct")
    # debug_dataset(dataset)
    # dataset = prepare_dataset_guard(example_folder="logs-trained/dataset_guard")
    # debug_dataset(dataset)
