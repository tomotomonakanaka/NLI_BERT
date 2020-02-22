import pandas as pd
import re
from nltk.util import ngrams
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)

class NgramDataset(Dataset):
    def __init__(self, data_path, ngram2id=None):
        df = pd.read_csv(data_path)

        # Train
        if ngram2id == None:
            self.ngram2id = {}
            self.ngrams_data = []
            self.num_ngrams = []
            ngram_count = 1 # unknown = 0
            for i in range(len(df.Sentence)):
                ngram_ids = []
                word_ids = []
                s = df.Sentence[i]
                s = s.lower()
                s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
                tokens = [token for token in s.split(" ") if token != ""]
                output = list(ngrams(tokens, 3))
                for ngram in output:
                    if ngram in self.ngram2id:
                        ngram_ids.append(self.ngram2id[ngram])
                    else:
                        self.ngram2id[ngram] = ngram_count
                        ngram_ids.append(self.ngram2id[ngram])
                        ngram_count += 1
                self.ngrams_data.append(ngram_ids)
                self.num_ngrams.append(len(ngram_ids))

        # Dev
        else:
            self.ngram2id = ngram2id
            self.ngrams_data = []
            self.num_ngrams = []
            for i in range(len(df.Sentence)):
                ngram_ids = []
                word_ids = []
                s = df.Sentence[i]
                s = s.lower()
                s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
                tokens = [token for token in s.split(" ") if token != ""]
                output = list(ngrams(tokens, 3))
                for ngram in output:
                    if ngram in self.ngram2id:
                        ngram_ids.append(self.ngram2id[ngram])
                    else:
                        ngram_ids.append(len(self.ngram2id)+1)
                self.ngrams_data.append(ngram_ids)
                self.num_ngrams.append(len(ngram_ids))



        self.targets = df.L1.tolist()
        self.texts = df.TextFile.tolist()
        self.proficiency = df.Proficiency.tolist()

    def __getitem__(self, i):
        return self.ngrams_data[i], self.targets[i], self.texts[i], self.proficiency[i], self.num_ngrams[i]

    def __len__(self):
        return len(self.ngrams_data)

def collate(batch):
    ngram = []
    for item in batch:
        ngram.append(torch.LongTensor(item[0]))
    ngram_tensor = pad_sequence(ngram,batch_first=True)

    target = torch.LongTensor([item[1] for item in batch])
    proficiency = torch.LongTensor([item[3] for item in batch])
    num = torch.LongTensor([item[4] for item in batch])
    text = [item[2] for item in batch]

    ngram_tensor, target, proficiency, num = map(
        lambda x: x.to(device),
        (ngram_tensor, target, proficiency, num),
    )

    return ngram_tensor, target, text, proficiency, num
