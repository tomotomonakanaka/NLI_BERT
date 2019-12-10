import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM_BERT(nn.Module):
    def __init__(self, modelPATH, input_size, hidden_size, output_size=11, batch_first=True, num_layers=1, bidirectional=True, dropout=0.0):
        super(LSTM_BERT, self).__init__()


        self.BERTmodel = torch.load(modelPATH)
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first).to(device)
        self.hidden2out = nn.Linear(hidden_size, output_size).to(device)
        self.logsoftmax = nn.LogSoftmax().to(device)

        # self.dropout = nn.Dropout(p=dropout)

        # softmax

    def forward(self,inputsTensor,segmentTensor,maskTensor,targetTensor,roop,length):
        hiddens = []
        targets = []
        for i in range(inputsTensor.shape[0]):
            inputs = inputsTensor[i][:length[i]]
            segment = segmentTensor[i][:length[i]]
            mask = maskTensor[i][:length[i]]
            target = targetTensor[i][:length[i]]
            hidden = self.BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
            # hiddens_i = False
            # num_roop = length[i]/self.inner_batch_size
            # for j in range(num_roop):
            #     inputs = inputsTensor[i][j*self.inner_batch_size:(j+1)*self.inner_batch_size]
            #     segment = segmentTensor[i][j*self.inner_batch_size:(j+1)*self.inner_batch_size]
            #     mask = maskTensor[i][j*self.inner_batch_size:(j+1)*self.inner_batch_size]
            #     target = targetTensor[i][j*self.inner_batch_size:(j+1)*self.inner_batch_size]
            #     hidden = self.BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
            #     if (torch.is_tensor(hiddens_i) == False):
            #         hiddens_i = hidden
            #     else:
            #         hiddens_i = torch.cat((hiddens_i,hidden),0)
            # if (length[i] > num_roop*self.inner_batch_size):
            #     inputs = inputsTensor[i][num_roop*self.inner_batch_size:length[i]]
            #     segment = segmentTensor[i][num_roop*self.inner_batch_size:length[i]]
            #     mask = maskTensor[i][num_roop*self.inner_batch_size:length[i]]
            #     target = targetTensor[i][num_roop*self.inner_batch_size:length[i]]
            #     hidden = self.BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
            #     if (torch.is_tensor(hiddens_i) == False):
            #         hiddens_i = hidden
            #     else:
            #         hiddens_i = torch.cat((hiddens_i,hidden),0)
            hiddens.append(hidden)
            targets.append(target[0])



        hiddens_tensor = pad_sequence(hiddens,batch_first=True)
        targets = torch.LongTensor(targets).to(device)
        hiddens_len = length

        hiddens_len_sorted, hiddens_idx = torch.sort(hiddens_len, descending=True)
        hiddens_sorted = hiddens_tensor.index_select(dim=0, index=hiddens_idx)
        _, hiddens_ori_idx = torch.sort(hiddens_idx)

        hiddens_packed = nn.utils.rnn.pack_padded_sequence(hiddens_sorted, hiddens_len_sorted, batch_first=True)
        hiddens_packed, (h, c) = self.lstm(hiddens_packed)
        lstm_hiddens = h[-1].index_select(dim=0, index=hiddens_ori_idx)

        outputs = self.hidden2out(lstm_hiddens)
        outputs = self.logsoftmax(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 11), targets)

        return outputs, loss, targets
