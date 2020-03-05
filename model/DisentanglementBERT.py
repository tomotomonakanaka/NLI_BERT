from transformers import BertPreTrainedModel,BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

# Entropy Loss
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum() # I want to maximize, so I delete -1
        return b

class BertForToefl(BertPreTrainedModel):
    def __init__(self, config, num_prompt):
        super(BertForToefl, self).__init__(config)

        # number of L1 class and Prompt
        self.num_labels = config.num_labels
        self.num_prompt = num_prompt

        # Pretrained BERT model
        self.bert = BertModel(config)
        # self.bert = nn.DataParallel(self.bert, device_ids=[0,1], output_device=0)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # classifier
        self.L1Classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.PromptClassifier = nn.Linear(config.hidden_size, self.num_prompt)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, labels_prompt=None, L1=True):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]
        pooled_output_for_task = outputs[1]

        if L1 == True:
            pooled_output = self.dropout(pooled_output)
            logits = self.L1Classifier(pooled_output)
            logits_prompt = self.PromptClassifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            # calculating loss
            loss_fct = CrossEntropyLoss()
            adversarial_loss_fct = HLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + 0.03*adversarial_loss_fct(logits_prompt.view(-1, self.num_prompt))

            outputs = (loss,) + outputs
            outputs = outputs + (pooled_output_for_task,)
            return outputs  # loss, logits, (hidden_states), (attentions), pooled_output

        else:
            logits = self.PromptClassifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            # calculating loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_prompt), labels_prompt.view(-1))

            outputs = (loss,) + outputs
            outputs = outputs + (pooled_output_for_task,)
            return outputs  # loss, logits, (hidden_states), (attentions), pooled_output
