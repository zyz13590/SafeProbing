import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model
import os
import argparse
import sys
from pathlib import Path
import torch
import random
import numpy as np


def set_all_seeds(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seeds()

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="../data/train.json", type=str)
parser.add_argument("--model_path", default="/datas/huggingface/Qwen2.5-7B-Instruct/", type=str)
parser.add_argument("--save_path", default="results", type=str)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--model_name", default="qwen", type=str)
parser.add_argument("--epochs", default=2, type=int)
parser.add_argument("--grad_accum_steps", type=int, default=2)
parser.add_argument("--beta", required=True, type=float)
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

from data_helper import MyDataset, DataLoader


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "mistral" in model_path.lower():
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


if __name__ == "__main__":
    base_model, tokenizer = load_model(args.model_path)

    data_path = args.data_path
    data = json.load(open(data_path, "r"))
    dataset = MyDataset(data, tokenizer, model_name=args.model_name, MAX_LENGTH=300)

    config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False, 
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    peft_model = get_peft_model(base_model, config)
    peft_model.print_trainable_parameters()

    from MyTrainer import MyTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        output_dir=f"{args.save_path}/beta{args.beta}",
        logging_steps=1,
        save_steps=50,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        remove_unused_columns=False,
        report_to="none",
        fp16=True,
        fp16_full_eval=True,
        gradient_accumulation_steps=args.grad_accum_steps,
        save_total_limit=1,
        # deepspeed="./ds_config.json",
    )
    trainer = MyTrainer(
        model=peft_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        args=training_args,
        beta=args.beta,
    )

    trainer.train()
