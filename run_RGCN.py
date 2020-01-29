from model.RGCN import *
import networkx as nx
import time
import numpy as np
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import pandas as pd
import pickle
from sklearn.metrics import classification_report

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print("device",device)

infile = open('graph','rb')
g1 = pickle.load(infile)
g1 = g1.to(device)
print(type(g1))
infile.close()

infile = open('graph_add_edge','rb')
g2 = pickle.load(infile)
g2 = g2.to(device)
print(type(g2))
infile.close()

infile = open('hiddens','rb')
features = pickle.load(infile)
features = features.to(device)
infile.close()

infile = open('targets','rb')
labels = pickle.load(infile)
labels = labels.to(device)
infile.close()

infile = open('train_mask','rb')
train_mask = pickle.load(infile)
# train_mask = th.tensor(train_mask)
# train_mask = train_mask.to(device)
infile.close()

infile = open('dev_mask','rb')
dev_mask = pickle.load(infile)
# train_mask = th.tensor(train_mask)
# train_mask = train_mask.to(device)
infile.close()

infile = open('test_mask','rb')
test_mask = pickle.load(infile)
# test_mask = th.Tensor(test_mask)
# test_mask = test_mask.to(device)
infile.close()

net = RGCN(g1,g2,768,128,11,1,F.relu,0.1).to(device)


def evaluate(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(50):
    if epoch >=3:
        t0 = time.time()

    net.train()
    logits = net(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[dev_mask], labels[dev_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)

    acc = evaluate(net, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))

# prediction of text
net.eval()
with th.no_grad():
    logits = net(features)
    logits = logits[test_mask]
    logits = logits.detach().cpu().numpy()

TEST_ROW_PATH = "TOEFL11/test.csv"
TEST_PATH = "TOEFL_sentence/test_sentence.csv"

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
        preA += np.array(logits[i]) #* len(test_sentence_predict[i])
    else:
        if preT!=None:
            ans.append(np.argmax(preA))
        preA = np.array(logits[i]) #* len(test_sentence_predict[i])
        preT = test_array_predict[i]
ans.append(np.argmax(preA))

print(classification_report(ans, y_row_true))
print(np.sum(ans==y_row_true)/len(ans))
