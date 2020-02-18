import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepSets(nn.Module):
    def __init__(self,dropout=0.0):
        super(DeepSets, self).__init__()
        self.fc1 = nn.Linear(768,384)
        self.fc2 = nn.Linear(384,100)
        self.fc3 = nn.Linear(100,11)
        self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hiddens, length, target):
        hiddens = torch.sum(hiddens,axis=2)/length
        hiddens = F.relu(self.fc1(hiddens))
        hiddens = self.dropout(hiddens)
        hiddens = F.relu(self.fc2(hiddens))
        hiddens = self.dropout(hiddens)
        hiddens = self.fc3(hiddens)
        outputs = self.LogSoftmax(hiddens)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 11), target)
        return outputs, loss, target
