import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from model.data import ToeflDataset, collate
tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)


# path
TRAIN_PATH = "TOEFL11/TOEFL11_for_EF_sentences.csv"
TEST_PATH = "EF/EF_test_sentences.csv"
TRAIN_ROW_PATH = "TOEFL11/TOEFL11_for_EF.csv"
TEST_ROW_PATH = "EF/test_EF.csv"
modelPATH = "save_model/BERTSentence1"
bert_name = "bert-base-uncased"

# define parameter
max_len = 128
batch_size = 1

# define loader
train_dataset = ToeflDataset(TRAIN_PATH, max_len, bert_name)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
test_dataset = ToeflDataset(TEST_PATH, max_len, bert_name)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

# get sentences
train_df_sentences = pd.read_csv(TRAIN_PATH)
train_array_sentences = train_df_sentences.TextFile.values
test_df_sentences = pd.read_csv(TEST_PATH)
test_array_sentences = test_df_sentences.TextFile.values

# get ground truth
train_df_answer = pd.read_csv(TRAIN_ROW_PATH)
train_array_L1 = train_df_answer.L1.values
test_df_answer = pd.read_csv(TEST_ROW_PATH)
test_array_L1 = test_df_answer.L1.values

# load model
model = torch.load(modelPATH)
model = model.to(device)

# prediction of paragraph
print("Start Prediction")
def get_hiddens(data_loader, data_array_sentences):
    model.eval()
    index = 0
    num = 0
    preT = None
    hiddens = []
    with torch.no_grad():
        for inputs, mask, segment, target, text, num_tokens, prompt in data_loader:
            hidden = model(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[4].detach().cpu().numpy()
            if preT != data_array_sentences[index]:
                if preT != None:
                    hiddens.append(hidden_sum/num)
                hidden_sum = hidden
                num = 1
                preT = data_array_sentences[index]
            else:
                hidden_sum += hidden
                num += 1
            index += 1
        hiddens.append(hidden_sum/num)

    hiddens = np.array(hiddens)
    hiddens = hiddens.reshape((-1, 768))
    return hiddens

train_hiddens = get_hiddens(train_loader, train_array_sentences)
test_hiddens = get_hiddens(test_loader, test_array_sentences)

import pickle
outfile = open('train_hiddens','wb')
pickle.dump(train_hiddens,outfile)
outfile.close()
outfile = open('test_hiddens','wb')
pickle.dump(test_hiddens,outfile)
outfile.close()


from sklearn.linear_model import LogisticRegression
X = train_hiddens
y = train_array_L1
X_test = test_hiddens
y_test = test_array_L1
clf = LogisticRegression(max_iter=15).fit(X, y)
print(clf.score(X_test, y_test))
