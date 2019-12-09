import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from model.data import ToeflDataset, collate
tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMDataset(Dataset):
    def __init__(self, data_path, max_len, bert_config):
        dataset = ToeflDataset(data_path, max_len, bert_config)
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate)

        # make hidden_dicts and target_dicts
        inputs_dicts = {}
        mask_dicts = {}
        segment_dicts = {}
        target_dicts = {}
        counts = {}
        self.max_counts = 0;
        for inputs, mask, segment, target, text in loader:
            if (text[0] not in inputs_dicts):
                inputs_dicts[text[0]] = []
                inputs_dicts[text[0]].append(inputs[0])
                mask_dicts[text[0]] = []
                mask_dicts[text[0]].append(mask[0])
                segment_dicts[text[0]] = []
                segment_dicts[text[0]].append(segment[0])
                target_dicts[text[0]] = []
                target_dicts[text[0]].append(target[0])
                counts[text[0]] = 1

            else:
                inputs_dicts[text[0]].append(inputs[0])
                mask_dicts[text[0]].append(mask[0])
                segment_dicts[text[0]].append(segment[0])
                target_dicts[text[0]].append(target[0])
                counts[text[0]] += 1

            if counts[text[0]] > self.max_counts:
                self.max_counts = counts[text[0]]

        self.inputs_list = []
        self.mask_list = []
        self.segment_list = []
        self.target_list = []
        self.counts_list = []

        # make hiddens and targets
        for text in inputs_dicts:
            self.inputs_list.append(inputs_dicts[text])
            self.mask_list.append(mask_dicts[text])
            self.segment_list.append(segment_dicts[text])
            self.target_list.append(target_dicts[text])
            self.counts_list.append(counts[text])

    def __getitem__(self, i):
        return self.inputs_list[i], self.mask_list[i], self.segment_list[i], self.target_list[i], self.max_counts-self.counts_list[i], self.counts_list[i]

    def __len__(self):
        return len(self.inputs_list)


def collate_LSTM(batch):
    item = batch[0]
    shape0 = item[0][0].shape
    shape1 = item[1][0].shape
    shape2 = item[2][0].shape
    shape3 = item[3][0].shape
    inputs = []
    mask = []
    segment = []
    target = []
    roop = []
    length = []
    for item in batch:
        for i in range(item[4]):
            item[0].append(torch.zeros(shape0, dtype=torch.long).to(device))
            item[1].append(torch.zeros(shape1, dtype=torch.long).to(device))
            item[2].append(torch.zeros(shape2, dtype=torch.long).to(device))
            item[3].append(torch.zeros(shape3, dtype=torch.long).to(device))
        inputs.append(torch.stack(item[0]))
        mask.append(torch.stack(item[1]))
        segment.append(torch.stack(item[2]))
        target.append(torch.stack(item[3]))
        roop.append(item[4])
        length.append(item[5])

    inputs = torch.stack(inputs)
    mask = torch.stack(mask)
    segment = torch.stack(segment)
    target = torch.stack(target)
    roop = torch.IntTensor(roop)
    length = torch.IntTensor(length)

    inputs, mask, segment, target, roop, length = map(
        lambda x: x.to(device),
        (inputs, mask, segment, target, roop, length),
    )

    return inputs, mask, segment, target, roop, length
