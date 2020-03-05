import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)

PromptDict = {'P1':0, 'P2':1, 'P3':2, 'P4':3, 'P5':4, 'P6':5, 'P7':6, 'P8':7}
class ToeflDataset(Dataset):
    def __init__(self, data_path, max_len, bert_config):
        df = pd.read_csv(data_path)

        self.tokenizer = BertTokenizer.from_pretrained(bert_config)
        df['tokenized_text'] = df.Sentence.progress_apply(self.tokenizer.tokenize)

        # Shorten to max length (Bert has a limit of 512); subtract two tokens for [CLS] and [SEP]
        df.loc[:, 'tokenized_text'] = df.tokenized_text.str[:max_len - 2]

        # Add Bert-specific beginning and end tokens
        df.loc[:, 'tokenized_text'] = df.tokenized_text.apply(
            lambda tokens: ['[CLS]'] + tokens + ['[SEP]'],
        )

        df['indexed_tokens'] = df.tokenized_text.progress_apply(
            self.tokenizer.convert_tokens_to_ids,
        )

        sequences = df.indexed_tokens.tolist()
        max_sequence_length = max(len(x) for x in sequences)

        self.inputs_lst, self.masks, self.segments, self.num_tokens = [], [], [], []
        for sequence in sequences:
            self.inputs_lst.append(sequence + (max_sequence_length - len(sequence)) * [0])
            self.masks.append(len(sequence) * [1] + (max_sequence_length - len(sequence)) * [0])
            self.segments.append(max_sequence_length * [0])
            self.num_tokens.append(len(sequence))

        self.targets = df.L1.tolist()
        self.texts = df.TextFile.tolist()
        self.prompt = df.Prompt.tolist()
        for i in range(len(self.prompt)):
            self.prompt[i] = PromptDict[self.prompt[i]]

    def __getitem__(self, i):
        return self.inputs_lst[i], self.masks[i], self.segments[i], self.targets[i], self.texts[i], self.num_tokens[i], self.prompt[i]

    def __len__(self):
        return len(self.inputs_lst)

def collate(batch):
    inputs = torch.LongTensor([item[0] for item in batch])
    mask = torch.LongTensor([item[1] for item in batch])
    segment = torch.LongTensor([item[2] for item in batch])
    target = torch.LongTensor([item[3] for item in batch])
    num_tokens = torch.LongTensor([item[5] for item in batch])
    prompt = torch.LongTensor([item[6] for item in batch])
    text = [item[4] for item in batch]

    inputs, mask, segment, target, num_tokens, prompt = map(
        lambda x: x.to(device),
        (inputs, mask, segment, target, num_tokens, prompt),
    )

    return inputs, mask, segment, target, text, num_tokens, prompt
