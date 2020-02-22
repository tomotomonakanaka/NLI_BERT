import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepSetsFineTuned(nn.Module):
    def __init__(self,hidden1=256, hidden2=32, dropout=0.1):
        super(DeepSetsFineTuned, self).__init__()
        self.fc1 = nn.Linear(768,hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.fc3 = nn.Linear(hidden2,11)
        self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hiddens, length, target):
        length = length.reshape([-1,1])
        hiddens = torch.sum(hiddens,axis=1)/length
        hiddens = F.relu(self.fc1(hiddens))
        hiddens = self.dropout(hiddens)
        hiddens = F.relu(self.fc2(hiddens))
        hiddens = self.dropout(hiddens)
        hiddens = self.fc3(hiddens)
        outputs = self.logsoftmax(hiddens)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 11), target)
        return outputs, loss, target
