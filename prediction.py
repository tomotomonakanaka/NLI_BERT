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
from model.BERT import BertForToefl

tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)


# path
TRAIN_PATH = "TOEFL_sentence/train_sentence.csv"
DEV_PATH = "TOEFL_sentence/dev_sentence.csv"
TEST_PATH = "TOEFL_sentence/test_sentence.csv"
TEST_ROW_PATH = "TOEFL11/test.csv"
modelPATH = "save_model/positionModel"


# define parameter
max_len = 128
batch_size = 32
max_epochs = 5
num_training_steps = max_epochs * int(19740/batch_size)
num_warmup_steps = int(num_training_steps*0.1)
bert_name = "bert-base-uncased"
learning_rate = 5e-5


# define loader
test_dataset = ToeflDataset(TEST_PATH, max_len, bert_name)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

model = torch.load(modelPATH)
model = model.to(device)

# prediction of paragraph
print("Start Prediction")
model.eval()
y_true, y_pred = [], []
y_logits = []
with torch.no_grad():
    for inputs, mask, segment, target, text, a, b, position in test_loader:
        loss,logits = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target, position=position)[:2]

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        target = target.cpu().numpy()

        y_true.extend(predictions)
        y_pred.extend(target)
        y_logits.extend(logits)
print(classification_report(y_pred, y_true))


'''max pooling'''
# prediction of text
test_df_true = pd.read_csv(TEST_ROW_PATH)
y_row_true = test_df_true.L1.values

test_df_predict = pd.read_csv(TEST_PATH)
test_array_predict = test_df_predict.TextFile.values
test_sentence_predict = test_df_predict.Sentence.values

preT = None
preA = np.array([0,0,0,0,0,0,0,0,0,0,0])
ans = []


for i in range(len(test_array_predict)):
    if preT == test_array_predict[i]:
        arg_i = np.argmax(y_logits[i])
        probability = y_logits[i][arg_i]
        if probability > max_probability:
            max_arg_i = arg_i
            max_probability = probability
    else:
        if preT!=None:
            ans.append(max_arg_i)
        arg_i = np.argmax(y_logits[i])
        probability = y_logits[i][arg_i]
        max_arg_i = arg_i
        max_probability = probability
        preT = test_array_predict[i]
ans.append(max_arg_i)

print(classification_report(ans, y_row_true))
print(np.sum(ans==y_row_true)/len(ans))


# prediction of text
# test_df_true = pd.read_csv(TEST_ROW_PATH)
# y_labels = test_df_true.L1.values
#
# test_df_predict = pd.read_csv(TEST_PATH)
# test_array_predict = test_df_predict.TextFile.values
# test_sentence_predict = test_df_predict.Sentence.values
# test_sentence_labels = test_df_predict.L1.values
#
#
#
# y_probabilityies = []
# y_args = []
# for i in range(len(test_array_predict)):
#     arg_i = np.argmax(y_logits[i])
#     y_args.append(arg_i)
#     y_prob = np.exp(y_logits[i][arg_i])
#     y_probabilityies.append(y_probabilityies)
#
# y_probabilityies = np.array(y_probabilityies)
# y_args = np.array(y_args)
#
# data = {"id":test_array_predict, "text":test_sentence_predict,"predict label":y_args,
#         "probability":y_probabilityies, "label":test_sentence_labels}
# dataframe = pd.DataFrame.from_dict(data)
# dataframe.to_csv('kakunin.csv')
# print(classification_report(ans, y_row_true))
# print(np.sum(ans==y_row_true)/len(ans))
