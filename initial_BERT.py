from model.BERT import BertForToefl
import torch
bert_name = "bert-base-uncased"
model = BertForToefl.from_pretrained(bert_name, num_labels=11,output_hidden_states=True)
torch.save(model, "save_model/initialBERT")
