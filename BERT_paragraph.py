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

from model.data import ToeflDataset, collate

tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)


# path
TRAIN_PATH = "TOEFL11/train_paragraph.csv"
DEV_PATH = "TOEFL11/dev_paragraph.csv"
TEST_PATH = "TOEFL11/test_paragraph.csv"
TEST_ROW_PATH = "TOEFL11/test.csv"


# define parameter
max_len = 220
batch_size = 16
max_epochs = 5
num_training_steps = max_epochs * int(50310/batch_size)
num_warmup_steps = int(num_training_steps*0.1)
bert_name = "bert-base-uncased"
learning_rate = 2e-5


# define loader
train_dataset = ToeflDataset(TRAIN_PATH, max_len, bert_name)
dev_dataset = ToeflDataset(DEV_PATH, max_len, bert_name)
test_dataset = ToeflDataset(TEST_PATH, max_len, bert_name)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate,shuffle='True')
valid_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)


# load model
print("Load Model")
model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=11)
model = model.to(device)


# define optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


# define training and validation
def train_epoch(model, optimizer, train_loader):
    model.train()
    train_loss = total = 0
    for inputs, mask, segment, target, text in tqdm(train_loader,
                                                             desc='Training',
                                                             leave=False):
        optimizer.zero_grad()
        loss = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[0]
        train_loss += loss.item()
        total += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return train_loss / total

def validate_epoch(model, valid_loader):
    model.eval()
    with torch.no_grad():
        valid_loss = total = 0
        for inputs, mask, segment, target, text in tqdm(valid_loader,
                                                                 desc='Validating',
                                                                 leave=False):
            loss = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[0]
            valid_loss += loss.item()
            total += 1
        return valid_loss / total


# training
print("Start Training!")
n_epochs = 0
train_losses, valid_losses = [], []
while True:
    train_loss = train_epoch(model, optimizer, train_loader)
    valid_loss = validate_epoch(model, valid_loader)
    tqdm.write(
        f'epoch #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.3f}\tvalid_loss: {valid_loss:.3f}\n',
    )
    # Early stopping if the current valid_loss is
    # greater than the last three valid losses
    if len(valid_losses) > 2 and all(valid_loss > loss for loss in valid_losses[-3:]):
        print('Stopping early')
        break

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    n_epochs += 1

    if n_epochs >= max_epochs:
        break


# save loss graph
epoch_ticks = range(1, n_epochs + 1)
plt.plot(epoch_ticks, train_losses)
plt.plot(epoch_ticks, valid_losses)
plt.legend(['Train Loss', 'Valid Loss'])
plt.title('Losses')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.xticks(epoch_ticks)
plt.savefig('loss.png')


# prediction of paragraph
print("Start Prediction")
model.eval()
y_true, y_pred = [], []
y_logits = []
with torch.no_grad():
    for inputs, mask, segment, target, text in test_loader:
        loss,logits = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[:2]

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        target = target.cpu().numpy()

        y_true.extend(predictions)
        y_pred.extend(target)
        y_logits.extend(logits)
print(classification_report(y_pred, y_true))



# prediction of text
test_df_true = pd.read_csv(TEST_ROW_PATH)
y_row_true = test_df_true.L1.values

test_df_predict = pd.read_csv(TEST_PATH)
test_array_predict = test_df_predict.TextFile.values

preT = None
preA = np.array([0,0,0,0,0,0,0,0,0,0,0])
ans = []
for i in range(len(test_array_predict)):
    if preT == test_array_predict[i]:
        preA += np.array(y_logits[i])
    else:
        if preT!=None:
            ans.append(np.argmax(preA))
        preA = np.array(y_logits[i])
        preT = test_array_predict[i]
ans.append(np.argmax(preA))

print(classification_report(ans, y_row_true))

model.save_pretrained('./pretrained')
