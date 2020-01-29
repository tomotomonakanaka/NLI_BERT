import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM_BERT(nn.Module):
    def __init__(self, modelPATH, input_size, hidden_size, output_size=11, batch_first=True, num_layers=1, bidirectional=True, dropout=0.0):
        super(LSTM_BERT, self).__init__()

        # load the model from model PATH
        self.BERTmodel = torch.load(modelPATH)

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
        for i in range(inputsTensor.shape[0]):
            inputs = inputsTensor[i][:length[i]]
            segment = segmentTensor[i][:length[i]]
            mask = maskTensor[i][:length[i]]
            target = targetTensor[i][:length[i]]
            hidden = self.BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
            hiddens.append(hidden)
            targets.append(target[0])

        hiddens_tensor = pad_sequence(hiddens,batch_first=True)
        targets = torch.LongTensor(targets).to(device)
        hiddens_len = length

        hiddens_len_sorted, hiddens_idx = torch.sort(hiddens_len, descending=True)
        hiddens_sorted = hiddens_tensor.index_select(dim=0, index=hiddens_idx)
        _, hiddens_ori_idx = torch.sort(hiddens_idx)

        hiddens_packed = nn.utils.rnn.pack_padded_sequence(hiddens_sorted, hiddens_len_sorted, batch_first=True)
        hiddens_packed = hiddens_packed.cuda()
        hiddens_packed, h = self.lstm(hiddens_packed)
        lstm_hiddens = h[-1].index_select(dim=0, index=hiddens_ori_idx)

        outputs = self.hidden2out(lstm_hiddens)
        outputs = self.logsoftmax(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 11), targets)

        return outputs, loss, targets
