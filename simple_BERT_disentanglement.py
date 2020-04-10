import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm, tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.data_prompt import ToeflDataset, collate
from model.DisentanglementBERT import BertForToefl

tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)

# path
TRAIN_PATH = "TOEFL11/train.csv"
DEV_PATH = "TOEFL11/dev.csv"
TEST_PATH = "TOEFL11/test.csv"
TEST_ROW_PATH = "TOEFL11/test.csv"
modelPATH = "save_model/paragraphdisentangsimpleModel"


# define parameter
max_len = 512
batch_size = 12
max_epochs = 4
num_training_steps = max_epochs * int(9900/batch_size)
num_warmup_steps = int(num_training_steps*0.1)
bert_name = "bert-base-uncased"
learning_rate_bert = 6e-5
learning_rate_L1 = 6e-5
learning_rate_prompt = 6e-5

# define loader
train_dataset = ToeflDataset(TRAIN_PATH, max_len, bert_name)
valid_dataset = ToeflDataset(DEV_PATH, max_len, bert_name)
test_dataset = ToeflDataset(TEST_PATH, max_len, bert_name)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate,shuffle='True')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

# load model
print("Load Model")
model = BertForToefl.from_pretrained(bert_name, num_labels=11,output_hidden_states=True, num_prompt=8)
model = model.to(device)

# define optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters_L1 = [
    {'params': [p for n, p in param_optimizer if 'L1Classifier' in n and 'weight' in n],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if 'L1Classifier' in n and 'bias' in n],
     'weight_decay_rate': 0.0}]
optimizer_grouped_parameters_bert = [
    {'params': [p for n, p in param_optimizer if 'bert' in n],
     'weight_decay_rate': 0.01}]
optimizer_grouped_parameters_prompt = [
    {'params': [p for n, p in param_optimizer if 'PromptClassifier' in n and 'weight' in n],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if 'PromptClassifier' in n and 'bias' in n],
     'weight_decay_rate': 0.0}]

optimizer_L1 = AdamW(optimizer_grouped_parameters_L1, lr=learning_rate_L1)
optimizer_bert = AdamW(optimizer_grouped_parameters_bert, lr=learning_rate_bert)
optimizer_prompt = AdamW(optimizer_grouped_parameters_prompt, lr=learning_rate_prompt)
scheduler_L1 = get_linear_schedule_with_warmup(optimizer_L1, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
scheduler_prompt = get_linear_schedule_with_warmup(optimizer_prompt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


# define training and validation
def train_epoch(model, train_loader):
    model.train()
    train_loss_L1 = 0
    train_loss_bert = 0
    train_loss_prompt = 0
    total = 0
    for inputs, mask, segment, target, text, num_tokens, prompt in tqdm(train_loader,
                                                             desc='Training',
                                                             leave=False):
        # the number of roop
        total += 1

        '''calculating L1'''
        for p in model.bert.parameters():
            p.requires_grad = False
        for p in model.L1Classifier.parameters():
            p.requires_grad = True
        for p in model.PromptClassifier.parameters():
            p.requires_grad = False
        optimizer_L1.zero_grad()
        optimizer_prompt.zero_grad()
        optimizer_bert.zero_grad()
        loss_L1 = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target, mode='L1')[0]
        train_loss_L1 += loss_L1.item()
        loss_L1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_L1.step()
        scheduler_L1.step()

        '''calculating prompt'''
        for p in model.bert.parameters():
            p.requires_grad = False
        for p in model.L1Classifier.parameters():
            p.requires_grad = False
        for p in model.PromptClassifier.parameters():
            p.requires_grad = True
        optimizer_L1.zero_grad()
        optimizer_prompt.zero_grad()
        optimizer_bert.zero_grad()
        loss_prompt = model(inputs, token_type_ids=segment, attention_mask=mask, labels_prompt=prompt, mode='prompt')[0]
        train_loss_prompt += loss_prompt.item()
        loss_prompt.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_prompt.step()
        scheduler_prompt.step()

        '''optimizing bert'''
        for p in model.bert.parameters():
            p.requires_grad = True
        for p in model.L1Classifier.parameters():
            p.requires_grad = False
        for p in model.PromptClassifier.parameters():
            p.requires_grad = False
        optimizer_L1.zero_grad()
        optimizer_prompt.zero_grad()
        optimizer_bert.zero_grad()
        loss_bert = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target, mode='bert')[0]
        train_loss_bert += loss_bert.item()
        loss_bert.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_bert.step()
        scheduler_bert.step()
    return train_loss_L1 / total, train_loss_prompt / total, train_loss_bert / total



def validate_epoch(model, valid_loader):
    model.eval()
    with torch.no_grad():
        valid_loss_L1 = 0
        valid_loss_bert = 0
        valid_loss_prompt = 0
        total = 0
        for inputs, mask, segment, target, text, num_tokens, prompt in tqdm(valid_loader,
                                                                 desc='Validating',
                                                                 leave=False):
            total += 1

            # L1
            loss = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target, mode='L1')[0]
            valid_loss_L1 += loss.item()

            # prompt
            loss = model(inputs, token_type_ids=segment, attention_mask=mask, labels_prompt=prompt, mode='prompt')[0]
            valid_loss_prompt += loss.item()

            # bert
            loss = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target, mode='bert')[0]
            valid_loss_bert += loss.item()

        return valid_loss_L1 / total, valid_loss_prompt / total, valid_loss_bert / total


# training
print("Start Training!")
n_epochs = 0
while True:
    train_loss_L1, train_loss_prompt, train_loss_bert = train_epoch(model, train_loader)
    valid_loss_L1, valid_loss_prompt, valid_loss_bert = validate_epoch(model, valid_loader)
    tqdm.write(
        f'epoch #{n_epochs + 1:3d}\ttrain_loss_L1: {train_loss_L1:.3f}\tvalid_loss_L1: {valid_loss_L1:.3f}\n',
    )
    tqdm.write(
        f'epoch #{n_epochs + 1:3d}\ttrain_loss_prompt: {train_loss_prompt:.3f}\tvalid_loss_prompt: {valid_loss_prompt:.3f}\n',
    )
    tqdm.write(
        f'epoch #{n_epochs + 1:3d}\ttrain_loss_bert: {train_loss_bert:.3f}\tvalid_loss_L1: {valid_loss_bert:.3f}\n',
    )

    n_epochs += 1

    if n_epochs >= max_epochs:
        break




torch.save(model, modelPATH)

# prediction of paragraph
print("Start Prediction")
model.eval()
y_true, y_pred = [], []
y_logits = []
y_true_prompt, y_pred_prompt = [], []
y_logits_prompt = []
with torch.no_grad():
    for inputs, mask, segment, target, text, num_tokens, prompt in test_loader:
        # L1
        loss,logits = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target, mode='L1')[:2]
        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        target = target.cpu().numpy()
        y_true.extend(target)
        y_pred.extend(predictions)
        y_logits.extend(logits)

        # prompt
        loss, logits_prompt = model(inputs, token_type_ids=segment, attention_mask=mask, labels_prompt=prompt, mode='prompt')[:2]
        logits_prompt = logits_prompt.detach().cpu().numpy()
        predictions_prompt = np.argmax(logits_prompt, axis=1)
        prompt = prompt.cpu().numpy()
        y_pred_prompt.extend(predictions_prompt)
        y_true_prompt.extend(prompt)
        y_logits_prompt.extend(logits_prompt)
print(classification_report(y_pred, y_true))
print(classification_report(y_pred_prompt, y_true_prompt))
