# NLI_MODELS
## Setup
```
$ unzip TOEFL11.zip
$ pip install pytorch # depending on your device
$ pip install -r requirements.txt
$ git clone https://github.com/huggingface/transformers
$ pip install transformers
```
## Run simple BERT model
```
$ python BERT_simple.py
```

### How can I improve
```
1. split data into paragraph  
-we can increase the datasets and decrease the parameters of BERT  
2. Using RoBERTa  
-now we are overfitting  
3. increasing dropout probabilities  
4. decreasing learning rate or changing warm up rate  
5. pretrain our model in task domain
```

### Results
max_len = 512  
batch_size = 4  
max_epochs = 5  
bert_name = "bert-base-uncased"  
learning_rate = 0.0001  
param_optimizer = list(model.classifier.named_parameters())

```
              precision    recall  f1-score   support

           0       0.10      0.32      0.15        31
           1       0.48      0.17      0.25       282
           2       0.05      0.23      0.08        22
           3       0.20      0.30      0.24        67
           4       0.26      0.27      0.27        96
           5       0.04      0.31      0.07        13
           6       0.12      0.44      0.19        27
           7       0.07      0.50      0.12        14
           8       0.63      0.17      0.27       370
           9       0.12      0.22      0.15        55
          10       0.32      0.26      0.29       123

    accuracy                           0.22      1100
   macro avg       0.22      0.29      0.19      1100
weighted avg       0.42      0.22      0.25      1100
```
max_len = 512  
batch_size = 4  
max_epochs = 4  
bert_name = "bert-base-cased"  
learning_rate = 2.0e-5   
param_optimizer = list(model.named_parameters())  
no_decay = ['bias', 'gamma', 'beta']  
optimizer_grouped_parameters = [  
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],  
     'weight_decay_rate': 0.01},  
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],  
     'weight_decay_rate': 0.0}  
]
warm_up = 0.1
```
Start Training!
epoch #  1      train_loss: 1.987       valid_loss: 1.444                                                                                                                                                   

epoch #  2      train_loss: 1.206       valid_loss: 1.343                                                                                                                                                   

epoch #  3      train_loss: 0.912       valid_loss: 0.986                                                                                                                                                   

epoch #  4      train_loss: 0.676       valid_loss: 1.284                                                                                                                                                   

epoch #  5      train_loss: 0.513       valid_loss: 1.320                                                                                                                                                   

epoch #  6      train_loss: 0.362       valid_loss: 1.709                                                                                                                                                   

Stopping early
Start Prediction
              precision    recall  f1-score   support

           0       0.63      0.83      0.72        76
           1       0.79      0.85      0.82        93
           2       0.83      0.57      0.68       145
           3       0.69      0.71      0.70        97
           4       0.69      0.91      0.78        76
           5       0.84      0.68      0.75       124
           6       0.71      0.66      0.68       108
           7       0.78      0.68      0.73       115
           8       0.74      0.76      0.75        97
           9       0.45      0.88      0.60        51
          10       0.77      0.65      0.71       118

    accuracy                           0.72      1100
   macro avg       0.72      0.74      0.72      1100
weighted avg       0.74      0.72      0.72      1100
```

```
Start Training!
epoch #  1      train_loss: 1.701       valid_loss: 1.119                                                                                                                                                   

epoch #  2      train_loss: 0.882       valid_loss: 1.031                                                                                                                                                   

epoch #  3      train_loss: 0.467       valid_loss: 1.072                                                                                                                                                   

Start Prediction
              precision    recall  f1-score   support

           0       0.70      0.65      0.68       107
           1       0.85      0.81      0.83       105
           2       0.74      0.73      0.74       101
           3       0.58      0.66      0.62        88
           4       0.88      0.77      0.82       114
           5       0.81      0.70      0.75       116
           6       0.67      0.72      0.69        93
           7       0.71      0.66      0.69       107
           8       0.67      0.70      0.68        96
           9       0.67      0.73      0.70        92
          10       0.68      0.84      0.75        81

    accuracy                           0.72      1100
   macro avg       0.72      0.73      0.72      1100
weighted avg       0.73      0.72      0.73      1100
```
