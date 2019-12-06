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

from model.LSTM_data import LSTMDataset

tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)


# path
TRAIN_PATH = "TOEFL11/test_paragraph.csv"
modelPATH = "save_model/paragraphModel"


# define parameter
max_len = 220
bert_name = "bert-base-uncased"
learning_rate = 2e-5

# define loader
train_dataset = LSTMDataset(TRAIN_PATH, max_len, bert_name, modelPATH)
print(train_dataset[0])
