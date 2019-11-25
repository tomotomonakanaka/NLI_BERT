import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
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
TRAIN_PATH = "TOEFL11/train.csv"
DEV_PATH = "TOEFL11/dev.csv"
TEST_PATH = "TOEFL11/test.csv"

# define parameter
max_len = 512
batch_size = 4
max_epochs = 5
bert_name = "bert-base-uncased"
learning_rate = 0.0001

# define loader
train_dataset = ToeflDataset(TRAIN_PATH, max_len, bert_name)
dev_dataset = ToeflDataset(DEV_PATH, max_len, bert_name)
test_dataset = ToeflDataset(TEST_PATH, max_len, bert_name)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate,shuffle='True')
valid_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

# load model
model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=11)
model = model.to(device)

# define optimizer
param_optimizer = list(model.classifier.named_parameters())
optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

# define training and calidation
def train_epoch(model, optimizer, train_loader):
    model.train()
    train_loss = total = 0
    for inputs, mask, segment, target, text in tqdm_notebook(train_loader,
                                                             desc='Training',
                                                             leave=False):
        optimizer.zero_grad()
        loss = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[0]
        train_loss += loss.item()
        total += 1
        loss.backward()
        optimizer.step()
    return train_loss / total

def validate_epoch(model, valid_loader):
    model.eval()
    with torch.no_grad():
        valid_loss = total = 0
        for inputs, mask, segment, target, text in tqdm_notebook(valid_loader,
                                                                 desc='Validating',
                                                                 leave=False):
            loss = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[0]
            valid_loss += loss.item()
            total += 1
        return valid_loss / total

# training
max_epochs = 5
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

# prediction
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, mask, segment, target, text in test_loader:
        loss = model(inputs, segment, mask, target)
        logits = model(inputs, segment, mask)

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        target = target.cpu().numpy()

        y_true.extend(predictions)
        y_pred.extend(target)

print(classification_report(y_true, y_pred))
