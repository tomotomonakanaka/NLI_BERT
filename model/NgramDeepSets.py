import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NgramModel(nn.Module):
    def __init__(self,vocab_size, embedding_dim=768, hidden1=128, hidden2=32, dropout=0.1):
        super(NgramDeepSets, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # MLP
        self.fc1 = nn.Linear(embedding_dim,hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.fc3 = nn.Linear(hidden2,11)
        self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,ngrams,num_ngram,target):
        hidden = self.embeddings(ngrams)

        # MLP
        hidden = torch.sum(hidden,axis=1) # 改善の余地
        hidden = F.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        hidden = self.fc3(hidden)
        outputs = self.logsoftmax(hidden)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 11), target)

        return outputs, loss, target
