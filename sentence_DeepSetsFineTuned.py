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
import pickle

from model.finetuned_BERT_data import *
from model.DeepSetsFineTuned import *

tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)



# path
TRAIN_PATH = "TOEFL_sentence/train_sentence.csv"
TRAIN_OUT = "TOEFL_sentence/train_out"
DEV_PATH = "TOEFL_sentence/dev_sentence.csv"
DEV_OUT = "TOEFL_sentence/dev_out"
TEST_PATH = "TOEFL_sentence/test_sentence.csv"
TEST_OUT = "TOEFL_sentence/test_out"
TEST_ROW_PATH = "TOEFL11_sentence/test.csv"
modelPATH = "save_model/paragraphModel"
make_dataloader = False

# define parameter
max_len = 128
batch_size = 16
max_epochs = 10
num_training_steps = max_epochs * int(1100/batch_size)
num_warmup_steps = int(num_training_steps*0.1)
bert_name = "bert-base-uncased"
learning_rate = 1e-3

# datasets and dataloaders
if make_dataloader==True:
    train_dataset = FinetunedData(TRAIN_PATH, max_len,bert_name, modelPAT)
    train_out = open(TRAIN_OUT,'wb')
    pickle.dump(train_dataset, train_out)
    train_out.close()
    valid_dataset = FinetunedData(DEV_PATH, max_len,bert_name, modelPATH)
    valid_out = open(DEV_OUT,'wb')
    pickle.dump(valid_dataset, valid_out)
    valid_out.close()
    test_dataset = FinetunedData(TEST_PATH, max_len,bert_name, modelPATH)
    test_out = open(TEST_OUT,'wb')
    pickle.dump(test_dataset, test_out)
    test_out.close()
else:
    infile = open(TRAIN_OUT,'rb')
    train_dataset = pickle.load(infile)
    infile.close()
    infile = open(DEV_OUT,'rb')
    valid_dataset = pickle.load(infile)
    infile.close()
    infile = open(TEST_OUT,'rb')
    test_dataset = pickle.load(infile)
    infile.close()


train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_LSTM, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_LSTM)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_LSTM)


# load model
print("Load Model")
model = DeepSetsFineTuned()
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
def train_epoch(model, optimizer, train_loader):
    model.train()
    count = 0
    train_loss = total = 0
    for hiddens, target, length, num_tokens in tqdm(train_loader,desc='Training',leave=False):
        if count == 0:
            print("hiddens shape: ", hiddens.shape)
            print("length shape: ", length.shape)
            count += 1
        optimizer.zero_grad()
        loss = model(hiddens, length, target)[1]
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
        for hiddens, target, length, num_tokens in tqdm(valid_loader,desc='Validating',leave=False):
            loss = model(hiddens, length, target)[1]
            valid_loss += loss.item()
            total += 1
        return valid_loss / total


# training
print("Start Training!")
n_epochs = 0
train_losses, valid_losses = [], []
while True:
    train_loss = train_epoch(model, optimizer, valid_loader)
    valid_loss = validate_epoch(model, test_loader)
    tqdm.write(
        f'epoch #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.3f}\tvalid_loss: {valid_loss:.3f}\n',
    )
    # Early stopping if the current valid_loss is
    # greater than the last three valid losses
    # if len(valid_losses) > 2 and all(valid_loss > loss for loss in valid_losses[-3:]):
    #     print('Stopping early')
    #     break

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
    for hiddens, target, length, num_tokens in test_loader:
        logits, loss, target = model(hiddens,length,target)[:]

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        target = target.cpu().numpy()

        y_true.extend(target)
        y_pred.extend(predictions)
        y_logits.extend(logits)
y_true = np.array(y_true)
print(np.sum(y_true==y_pred)/len(y_true))
print(batch_size)
print(classification_report(y_pred, y_true))
