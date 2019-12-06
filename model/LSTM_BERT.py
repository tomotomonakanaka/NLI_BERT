import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_BERT(nn.Module):
    def __init__(self, modelPATH, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=True, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.BERTmodel = torch.load(modelPATH)
        self.dropout = nn.Dropout(p=dropout)

        # softmax

    def forward(self,inputsTensor,segmentTensor,maskTensor,targetTensor,length):
        hiddens = []
        for i in inputsTensor.shape[0]:
            inputs = inputsTensor[i]
            segment = segmentTensor[i]
            mask = maskTensor[i]
            target = targetTensor[i]
            hidden = self.BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[2]
            hiddens.append(hidden)

        hiddens_len = length
        hiddens = torch.Tensor(hiddens)
        hiddens = self.dropout(hiddens)

        # rnn
        # softmax


        return x, h
