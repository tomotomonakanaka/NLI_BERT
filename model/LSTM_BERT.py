import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from model.BERT import BertForToefl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM_BERT(nn.Module):
    def __init__(self, modelPATH, input_size, hidden_size, output_size=11, batch_first=True, num_layers=1, bidirectional=True, dropout=0.0):
        super(LSTM_BERT, self).__init__()

        # load the model from model PATH
        self.BERTmodel = BertForToefl.from_pretrained(modelPATH, num_labels=11,output_hidden_states=True)

        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first,
                           dropout = dropout).to(device)

        # other layers
        self.dropout = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_size, output_size).to(device)
        self.logsoftmax = nn.LogSoftmax().to(device)

    def forward(self,inputsTensor,segmentTensor,maskTensor,targetTensor,roop,length):
        hiddens = []
        targets = []
        inputs = inputsTensor[0][:length[0]]
        segment = segmentTensor[0][:length[0]]
        mask = maskTensor[0][:length[0]]
        target = targetTensor[0][:length[0]]
        hidden = self.BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
        hidden = hidden.view(1,-1,768)
        hidden, (h, c) = self.lstm(hidden)
        lstm_hiddens = h[-1]
        outputs = self.hidden2out(lstm_hiddens)
        outputs = self.logsoftmax(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, torch.LongTensor([target[0]]).cuda())

        return outputs, loss, targets
