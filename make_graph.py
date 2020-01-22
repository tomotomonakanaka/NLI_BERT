from torch.utils.data import Dataset, DataLoader
import torch
from model.data import ToeflDataset, collate
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
from dgl import DGLGraph
import pickle
tqdm.pandas()


# make hidden_dicts and target_dicts
hiddens = []
hiddens_tensor = []
positions = []
targets = []
texts = []
train_mask = []
dev_mask = []
test_mask = []
max_len = 128
batch_size = 1
bert_name = "bert-base-uncased"
modelPATH = "save_model/paragraphModel"
TRAIN_PATH = "TOEFL_sentence/train_sentence.csv"
DEV_PATH = "TOEFL_sentence/dev_sentence.csv"
TEST_PATH = "TOEFL_sentence/test_sentence.csv"

BERTmodel = torch.load(modelPATH)
BERTmodel.eval()

# define loader
train_dataset = ToeflDataset(TRAIN_PATH, max_len, bert_name)
valid_dataset = ToeflDataset(DEV_PATH, max_len, bert_name)
test_dataset = ToeflDataset(TEST_PATH, max_len, bert_name)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

with torch.no_grad():
    for inputs, mask, segment, target, text, proficiency, num_tokens, position in tqdm(train_loader,desc='model_output',leave=False):
        hidden = BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
        hiddens.append(hidden.tolist()[0])
        hiddens_tensor.append(hidden[0])
        targets.append(target[0])
        texts.append(text[0])
        train_mask.append(True)
        dev_mask.append(False)
        test_mask.append(False)
    for inputs, mask, segment, target, text, proficiency, num_tokens, position in tqdm(valid_loader,desc='model_output',leave=False):
        hidden = BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
        hiddens.append(hidden.tolist()[0])
        hiddens_tensor.append(hidden[0])
        targets.append(target[0])
        texts.append(text[0])
        train_mask.append(False)
        dev_mask.append(True)
        test_mask.append(False)
    for inputs, mask, segment, target, text, proficiency, num_tokens, position in tqdm(test_loader,desc='model_output',leave=False):
        hidden = BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
        hiddens.append(hidden.tolist()[0])
        hiddens_tensor.append(hidden[0])
        targets.append(target[0])
        texts.append(text[0])
        train_mask.append(False)
        dev_mask.append(False)
        test_mask.append(True)


A = kneighbors_graph(hiddens, 30, mode='connectivity', include_self=True)
g = DGLGraph()
g.from_scipy_sparse_matrix(A)
print("node num", g.number_of_nodes())
print("edge num", g.number_of_edges())

file_name = 'graph'
outfile = open(file_name, 'wb')
pickle.dump(g,outfile)
outfile.close()

hiddens_tensor = torch.stack(hiddens_tensor)
file_name = 'hiddens'
outfile = open(file_name, 'wb')
pickle.dump(hiddens_tensor,outfile)
outfile.close()

targets_tensor = torch.stack(targets)
file_name = 'targets'
outfile = open(file_name, 'wb')
pickle.dump(targets_tensor,outfile)
outfile.close()

file_name = 'texts'
outfile = open(file_name, 'wb')
pickle.dump(texts,outfile)
outfile.close()

file_name = 'train_mask'
outfile = open(file_name, 'wb')
pickle.dump(train_mask,outfile)
outfile.close()

file_name = 'dev_mask'
outfile = open(file_name, 'wb')
pickle.dump(dev_mask,outfile)
outfile.close()

file_name = 'test_mask'
outfile = open(file_name, 'wb')
pickle.dump(test_mask,outfile)
outfile.close()
