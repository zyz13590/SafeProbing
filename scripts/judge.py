import os
import argparse
from tqdm import tqdm
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True, help="Single prompt text to judge")
parser.add_argument("--response", default=None, help="Optional response text to judge")
parser.add_argument("--device", required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = os.environ["llama_guard"] # please see https://huggingface.co/meta-llama/Llama-Guard-3-8B for more details
dtype = torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")


def judge_single(prompt, response=None):
    if response is not None:
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    else:
        chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0, do_sample=False)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


if __name__ == "__main__":
    res = judge_single(args.prompt, args.response)
    print(f"Judgment result: {res}")
