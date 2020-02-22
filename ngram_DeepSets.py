import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm, tqdm_notebook
import torch
from torch.utils.data import DataLoader

from model.ngram_data import NgramDataset, collate

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
max_epochs = 4
learning_rate = 6e-5

# define dataset
train_dataset = NgramDataset(TRAIN_PATH)
dev_dataset = NgramDataset(DEV_PATH, train_dataset.ngram2id)
test_dataset = NgramDataset(TEST_PATH, train_dataset.ngram2id)
