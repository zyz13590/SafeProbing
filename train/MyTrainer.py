from transformers import Trainer, default_data_collator
import torch
from torch.utils.data import Dataset
from typing import Optional
import numpy as np
import random

from tqdm import tqdm
import torch.nn.functional as F
import math
import torch.nn as nn


loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)


class CustomValueLoss(nn.Module):
    def __init__(self, beta):

        super(CustomValueLoss, self).__init__()
        self.beta = beta
        self.mse_loss = nn.MSELoss()  

    def forward(self, values, labels):

        values = 2 / (1 + torch.exp(self.beta * values))
        loss = self.mse_loss(values, labels)

        return loss.mean()


class MyTrainer(Trainer):
    def __init__(self, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = CustomValueLoss(beta)

    def compute_loss(self, peft_model, batch, **k):
        input_ids = batch["input_ids"].to(peft_model.device)
        attention_mask = batch["attention_mask"].to(peft_model.device)
        labels = batch["labels"].to(peft_model.device)
        types = batch["type"].float().to(peft_model.device)
        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        mask = shifted_labels.ne(-100).float()
        per_token_loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)).view(shifted_labels.shape)
        token_losses = per_token_loss * mask
        per_sample_loss = token_losses.sum(dim=1)  
        per_sample_loss = per_sample_loss / mask.sum(dim=1)

        final_loss = torch.tensor(0.0, device=peft_model.device)
        sft_loss = 0.0
        if (types == -1).sum() != 0:
            sft_loss = (per_sample_loss * (types == -1)).sum() / (types == -1).sum()
            final_loss += 0.05 * sft_loss

        loss = per_sample_loss[types != -1]
        types = types[types != -1]
        monitor_loss = self.custom_loss(loss, types)

        final_loss += monitor_loss
        return final_loss
