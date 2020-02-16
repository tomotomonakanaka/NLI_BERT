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

from model.LSTM_data import *
from model.maxpooling_BERT import *

tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)



# path
TRAIN_PATH = "TOEFL_sentence/train_sentence.csv"
DEV_PATH = "TOEFL_sentence/dev_sentence.csv"
TEST_PATH = "TOEFL_sentence/test_sentence.csv"
modelPATH = "save_model/initialBERT"
savePATH = "save_model/LSTMModel"

# define parameter
max_len = 128
batch_size = 8
max_epochs = 8
num_training_steps = max_epochs * int(9900/batch_size)
num_warmup_steps = int(num_training_steps*0.1)
bert_name = "bert-base-uncased"
learning_rate = 6e-5
cls_hidden_size = 768
LSTM_hidden_size = 768

# define loader
train_dataset = LSTMDataset(TRAIN_PATH, max_len,bert_name)
train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_LSTM, shuffle='True')
valid_dataset = LSTMDataset(DEV_PATH, max_len,bert_name)
valid_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate_LSTM)
test_dataset = LSTMDataset(TEST_PATH, max_len,bert_name)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_LSTM)


# load model
print("Load Model")
model = MAX_BERT(bert_name)
model = model.to(device)

# define optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
for n,p in param_optimizer:
    print(n)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


# define training and validation
def train_epoch(model, optimizer, train_loader, batch_size):
    model.train()
    train_loss = 0
    total = 0
    optimizer.zero_grad()
    for inputs, mask, segment, target, roop, length in tqdm(train_loader,
                                                             desc='Training',
                                                             leave=False):
        loss = model(inputs, segment, mask, target, roop, length)[1]
        train_loss += loss.detach().item()
        total += 1
        loss.backward()

        # accumulation
        if total % batch_size == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return train_loss / total

def validate_epoch(model, valid_loader):
    model.eval()
    with torch.no_grad():
        valid_loss = total = 0
        for inputs, mask, segment, target, roop, length in tqdm(valid_loader,
                                                                 desc='Validating',
                                                                 leave=False):
            loss = model(inputs, segment, mask, target, roop, length)[1]
            valid_loss += loss.detach().item()
            total += 1
        return valid_loss / total


# training
print("Start Training!")
n_epochs = 0
train_losses, valid_losses = [], []

count = 0
while True:
    train_loss = train_epoch(model, optimizer, train_loader, batch_size)
    valid_loss = validate_epoch(model, valid_loader)
    torch.save(model, savePATH+"{}".format(count))
    count+=1
    tqdm.write(
        f'epoch #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.3f}\tvalid_loss: {valid_loss:.3f}\n',
    )
    Early stopping if the current valid_loss is
    greater than the last three valid losses
    if len(valid_losses) > 2 and all(valid_loss > loss for loss in valid_losses[-3:]):
        print('Stopping early')
        break

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    n_epochs += 1

    if n_epochs >= max_epochs:
        break


torch.save(model, savePATH)

# prediction of paragraph
print("Start Prediction")
model.eval()
y_true, y_pred = [], []
y_logits = []
with torch.no_grad():
    for inputs, mask, segment, target, roop, length in test_loader:
        logits, loss, targets = model(inputs, segment, mask, target, roop, length)[:]

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        target = target[0][0]

        y_true.append(target)
        y_pred.extend(predictions)
        y_logits.extend(logits)
y_true = np.array(y_true)
print(np.sum(y_true==y_pred)/len(y_true))
print(classification_report(y_pred, y_true))
