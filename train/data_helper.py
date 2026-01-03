from torch.utils.data import Dataset, DataLoader
import torch


class MyDataset(Dataset):
    def __init__(self, data: list, tokenizer, model_name="qwen", MAX_LENGTH=256):
        super().__init__()

        self.data = data
        self.MAX_LENGTH = MAX_LENGTH
        self.tokenizer = tokenizer
        self.model_name = model_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = self.data[index]["prompt"]
        response = self.data[index]["response"]
        harmful = self.data[index]["harmful"]
        ultrafeedback = self.data[index]["ultrafeedback"]
        chat = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
        input_len = len(chat)
        chat = self.tokenizer.decode(chat)
        if ultrafeedback:
            chat += response
            chat = self.tokenizer.decode(self.tokenizer.encode(chat, add_special_tokens=False)[: self.MAX_LENGTH])
            inputs = self.tokenizer(chat, padding="max_length", max_length=self.MAX_LENGTH, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[:input_len] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            chat += response
            chat = self.tokenizer.decode(self.tokenizer.encode(chat, add_special_tokens=False)[: self.MAX_LENGTH - 10])
            len1 = len(self.tokenizer.encode(chat, add_special_tokens=False))
            chat += " Note that this is illegal and unethical"
            inputs = self.tokenizer(chat, padding="max_length", max_length=self.MAX_LENGTH, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[:len1] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "type": -1 if ultrafeedback else harmful,
        }


if __name__ == "__main__":
    import json, os
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/datas/huggingface/Qwen2.5-7B-Instruct/", padding_side="right")
    # tokenizer = AutoTokenizer.from_pretrained("/datas/huggingface/Mistral-7B-Instruct-v0.3/", padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = MyDataset(
        data=json.load(open("../data/train.json", "r")),
        tokenizer=tokenizer,
        MAX_LENGTH=256,
        model_name="qwen",
    )
    import random

    print(len(dataset))
    item = dataset[10]
    labels = item["labels"]
    ids = item["input_ids"]
    print("-" * 50)
    print(tokenizer.decode(ids[ids != -100]))
    print("-" * 50)
    print(tokenizer.decode(labels[labels != -100]))
    print("-" * 50)
    print(item["type"])
