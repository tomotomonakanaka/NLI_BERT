import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_tokens_size, output_size=11, batch_first=True, num_layers=1, bidirectional=True, dropout=0.1):
        super(LSTM, self).__init__()
        # self.embedding = nn.Embedding(num_tokens_size+1,40)
        self.lstm = nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first,
                           dropout = dropout).to(device)
        self.hidden2out = nn.Linear(hidden_size, output_size).to(device)
        self.logsoftmax = nn.LogSoftmax().to(device)

        self.dropout = nn.Dropout(p=dropout)

        # softmax

    def forward(self, hiddens,target,length,num_tokens):
        # num_tokens = self.embedding(num_tokens)
        # hiddens = torch.cat((hiddens, num_tokens),2)
        # hiddens = hiddens.to(device)
        hiddens = self.dropout(hiddens)
        hiddens_len_sorted, hiddens_idx = torch.sort(length, descending=True)
        hiddens_sorted = hiddens.index_select(dim=0, index=hiddens_idx)
        _, hiddens_ori_idx = torch.sort(hiddens_idx)
        hiddens_packed = nn.utils.rnn.pack_padded_sequence(hiddens_sorted, hiddens_len_sorted, batch_first=True)
        # hiddens_packed, (h, c) = self.lstm(hiddens_packed)
        hiddens_packed, h = self.lstm(hiddens_packed)
        lstm_hiddens = h[-1].index_select(dim=0, index=hiddens_ori_idx)

        outputs = self.hidden2out(lstm_hiddens)
        outputs = self.logsoftmax(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 11), target)

        return outputs, loss, target
