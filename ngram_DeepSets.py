import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm, tqdm_notebook
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from model.ngram_data import NgramDataset, collate
from NgramDeepSets import NgramModel

tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)


# path
TRAIN_PATH = "TOEFL11/train.csv"
DEV_PATH = "TOEFL11/dev.csv"
TEST_PATH = "TOEFL11/test.csv"
modelPATH = "save_model/ngramModel"


# define parameter
batch_size = 32
max_epochs = 8
learning_rate = 6e-5

# define dataset
train_dataset = NgramDataset(TRAIN_PATH)
dev_dataset = NgramDataset(DEV_PATH, train_dataset.ngram2id)
test_dataset = NgramDataset(TEST_PATH, train_dataset.ngram2id)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate,shuffle='True')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)


# load model
print("Load Model")
model = NgramModel(len(train_dataset.ngram2id)+2)
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
    for ngrams, target, text, proficiency, num in tqdm(train_loader,
                                                             desc='Training',
                                                             leave=False):
        loss = model(ngrams, num, target)[1]
        train_loss += loss.item()
        total += 1
        loss.backward()

        if total % batch_size == 0: # accumulation
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
        for ngrams, target, text, proficiency, num in tqdm(valid_loader,
                                                                 desc='Validating',
                                                                 leave=False):
            loss = model(ngrams, num, target)[1]
            valid_loss += loss.item()
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
    count+=1
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



# prediction of paragraph
print("Start Prediction")
model.eval()
y_true, y_pred = [], []
y_logits = []
with torch.no_grad():
    for ngrams, target, text, proficiency, num in valid_loader:
        logits, loss, targets = model(ngrams, num, target)[:]

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        target = target[0][0]

        y_true.append(target)
        y_pred.extend(predictions)
        y_logits.extend(logits)
y_true = np.array(y_true)
print(np.sum(y_true==y_pred)/len(y_true))
print(classification_report(y_pred, y_true))
