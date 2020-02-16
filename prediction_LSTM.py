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
from model.LSTM_BERT import *

tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)

# path
TRAIN_PATH = "TOEFL_sentence/train_sentence.csv"
DEV_PATH = "TOEFL_sentence/dev_sentence.csv"
TEST_PATH = "TOEFL_sentence/test_sentence.csv"
modelPATH = "save_model/initialBERT"
savePATH = "save_model/LSTMModel3"

# define parameter
max_len = 128
batch_size = 16
max_epochs = 4
num_training_steps = max_epochs * int(9900/batch_size)
num_warmup_steps = int(num_training_steps*0.1)
bert_name = "bert-base-uncased"
learning_rate = 6e-5
cls_hidden_size = 768
LSTM_hidden_size = 100

# define loader
test_dataset = LSTMDataset(TEST_PATH, max_len,bert_name)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_LSTM)

model = torch.load(savePATH)
model = model.to(device)

# prediction of paragraph
print("Start Prediction")
model.eval()
y_true, y_pred = [], []
y_logits = []
i = 0
with torch.no_grad():
    for inputs, mask, segment, target, roop, length in tqdm(test_loader):
        logits, loss, targets = model(inputs, segment, mask, target, roop, length)[:]

        logits = logits.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        target = target[0][0]

        y_true.append(target)
        y_pred.extend(predictions)
        y_logits.extend(logits)
y_true = np.array(y_true)
print(np.sum(y_true==y_pred)/len(y_true))
print(classification_report(y_pred, y_true))
