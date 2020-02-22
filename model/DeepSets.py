import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BERT import BertForToefl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepSets(nn.Module):
    def __init__(self,modelPATH,multitask=False, hidden1=128, hidden2=32, dropout=0.1):
        super(DeepSets, self).__init__()

        # load the model from model PATH
        if multitask:
            self.BERTmodel = torch.load(modelPATH)
        else:
            self.BERTmodel = BertForToefl.from_pretrained(modelPATH, num_labels=11,output_hidden_states=True)

        # Using multi GPU
        self.BERTmodel = nn.DataParallel(self.BERTmodel, device_ids=[0,1], output_device=0)

        # MLP
        self.fc1 = nn.Linear(768,hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.fc3 = nn.Linear(hidden2,11)
        self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,inputsTensor,segmentTensor,maskTensor,targetTensor,roop,length):
        # get [CLS] from BERT
        inputs = inputsTensor[0][:length[0]]
        segment = segmentTensor[0][:length[0]]
        mask = maskTensor[0][:length[0]]
        target = targetTensor[0][:length[0]]
        hidden = self.BERTmodel(inputs, token_type_ids=segment, attention_mask=mask, labels=target)[3]
        target = torch.LongTensor([target[0]]).cuda()
        hidden = hidden.view(1,-1,768)

        # MLP
        hidden = torch.sum(hidden,axis=1)/length
        hidden = F.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        hidden = self.fc3(hidden)
        outputs = self.logsoftmax(hidden)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 11), target)

        return outputs, loss, target
