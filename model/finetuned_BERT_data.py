import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from model.data import ToeflDataset, collate
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FinetunedData(Dataset):
    def __init__(self, data_path, max_len, bert_config,modelPATH):

        self.BERTmodel = torch.load(modelPATH)
        self.BERTmodel.eval()

        dataset = ToeflDataset(data_path, max_len, bert_config)
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate)

        # make hidden_dicts and target_dicts
        hiddens_dicts = {}
        num_tokens_dicts = {}
        target_dicts = {}
        counts = {}
        self.max_counts = 0;
        with torch.no_grad():
            for inputs, mask, segment, target, text, proficiency, num_tokens in tqdm(loader,desc='model_output',leave=False):
                hidden = self.BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
                if (text[0] not in hiddens_dicts):
                    hiddens_dicts[text[0]] = []
                    hiddens_dicts[text[0]].append(hidden[0])
                    num_tokens_dicts[text[0]] = []
                    num_tokens_dicts[text[0]].append(num_tokens[0])
                    target_dicts[text[0]] = []
                    target_dicts[text[0]].append(target[0])
                    counts[text[0]] = 1

                else:
                    hiddens_dicts[text[0]].append(hidden[0])
                    num_tokens_dicts[text[0]].append(num_tokens[0])
                    target_dicts[text[0]].append(target[0])
                    counts[text[0]] += 1

                if counts[text[0]] > self.max_counts:
                    self.max_counts = counts[text[0]]

        self.hiddens_tensor = []
        self.num_tokens_tensor = []
        self.target_list = []
        self.counts_list = []

        # make hiddens and targets
        for text in hiddens_dicts:
            self.hiddens_tensor.append(torch.stack(hiddens_dicts[text]))
            self.num_tokens_tensor.append(torch.stack(num_tokens_dicts[text]))
            self.target_list.append(target_dicts[text])
            self.counts_list.append(counts[text])

    def __getitem__(self, i):
        return self.hiddens_tensor[i], self.target_list[i], self.counts_list[i], self.num_tokens_tensor[i]

    def __len__(self):
        return len(self.hiddens_tensor)


def collate_LSTM(batch):
    hiddens = []
    num_tokens = []
    target = []
    roop = []
    length = []
    hiddens_list = []
    for item in batch:
        hiddens.append(item[0])
        num_tokens.append(item[3])
        target.append(item[1][0])
        length.append(item[2])

    hiddens_tensor = pad_sequence(hiddens,batch_first=True)
    num_tokens_tensor = pad_sequence(num_tokens,batch_first=True)
    target = torch.LongTensor(target)
    length = torch.IntTensor(length)

    hiddens_tensor,target, length, num_tokens_tensor = map(
        lambda x: x.to(device),
        (hiddens_tensor, target, length, num_tokens_tensor),
    )

    return hiddens_tensor, target, length, num_tokens_tensor
