# NLI_MODELS
## Setup
```
$ unzip TOEFL11.zip
$ pip install pytorch # depending on your device
$ pip install -r requirements.txt
$ git clone https://github.com/huggingface/transformers
$ pip install transformers==2.2.1
```
## Simple BERT model
```
$ python BERT_simple.py
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

### How can I improve
```
1. split data into paragraph  
- we can increase the datasets and cover the all sentences in text.
- some text has more than 512 tokens
2. Using RoBERTa or AlBERT  
-now we are overfitting, so decrease the parameters  
3. changing hyperparameters (dropping out, learning rate, and warmup rate)  
4. pretrain our model in task domain
- https://arxiv.org/abs/1905.05583
5. usin dev+train after deciding hyperparameters
6. changing the way of sum up paragraph probability method
7. using bert-large
8. resizing paragraph model for not 220
```

## Paragraph Splitting BERT model
```
$ python BERT_paragraph.py
```

#### Back Ground
```
・If I want to classify Texts by contents, we have to consider all of the sentences in the Texts
- This is because the topic of text is at the beginning or ending
- deviding text decrease the accuracy of classification.
・However, if I want to classify the text by the structure, we don't have to consider it.
```
max_len = 220
batch_size = 16
max_epochs = 5
bert_name = "bert-base-uncased"  
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
epoch #  1      train_loss: 1.835       valid_loss: 1.474

epoch #  2      train_loss: 1.201       valid_loss: 1.273   

epoch #  3      train_loss: 0.814       valid_loss: 1.236                    

epoch #  4      train_loss: 0.543       valid_loss: 1.326        

epoch #  5      train_loss: 0.378       valid_loss: 1.422         

Stopping early
Start Prediction
              precision    recall  f1-score   support

           0       0.52      0.56      0.54       457
           1       0.62      0.70      0.66       582
           2       0.62      0.57      0.60       526
           3       0.56      0.60      0.58       530
           4       0.68      0.64      0.66       569
           5       0.59      0.64      0.61       451
           6       0.58      0.48      0.53       460
           7       0.54      0.51      0.52       502
           8       0.68      0.60      0.63       566
           9       0.50      0.57      0.53       458
          10       0.63      0.61      0.62       485

    accuracy                           0.59      5586
   macro avg       0.59      0.59      0.59      5586
weighted avg       0.60      0.59      0.59      5586

              precision    recall  f1-score   support

           0       0.90      0.82      0.86       110
           1       0.96      0.86      0.91       111
           2       0.76      0.92      0.83        83
           3       0.83      0.73      0.78       114
           4       0.88      0.91      0.89        97
           5       0.85      0.81      0.83       105
           6       0.76      0.86      0.81        88
           7       0.85      0.77      0.81       111
           8       0.76      0.82      0.79        93
           9       0.82      0.82      0.82       100
          10       0.79      0.90      0.84        88

    accuracy                           0.83      1100
   macro avg       0.83      0.84      0.83      1100
weighted avg       0.84      0.83      0.83      1100

accuracy: 0.8254545454545454
```

Using train + dev
```
Start Prediction
              precision    recall  f1-score   support

           0       0.52      0.59      0.55       457
           1       0.63      0.70      0.66       582
           2       0.62      0.56      0.59       526
           3       0.55      0.57      0.56       530
           4       0.69      0.66      0.67       569
           5       0.61      0.60      0.61       451
           6       0.57      0.51      0.54       460
           7       0.53      0.56      0.55       502
           8       0.67      0.63      0.65       566
           9       0.54      0.56      0.55       458
          10       0.67      0.63      0.65       485

    accuracy                           0.60      5586
   macro avg       0.60      0.60      0.60      5586
weighted avg       0.60      0.60      0.60      5586

              precision    recall  f1-score   support

           0       0.86      0.78      0.82       110
           1       0.95      0.84      0.89       113
           2       0.77      0.92      0.84        84
           3       0.79      0.71      0.75       111
           4       0.88      0.90      0.89        98
           5       0.87      0.86      0.87       101
           6       0.77      0.86      0.81        90
           7       0.83      0.78      0.80       107
           8       0.77      0.80      0.79        96
           9       0.81      0.87      0.84        93
          10       0.87      0.90      0.88        97

    accuracy                           0.83      1100
   macro avg       0.83      0.84      0.83      1100
weighted avg       0.84      0.83      0.83      1100

accuracy: 0.8336363636363636
```

## Paragraph Splitting and LSTM BERT model
```
epoch #  1      train_loss: 0.194       valid_loss: 0.637
epoch #  2      train_loss: 0.028       valid_loss: 0.694             
epoch #  3      train_loss: 0.024       valid_loss: 0.683 

Training:   0%|                                                                                                                                                                     | 0/310 [00:00<?, ?it/s]torch.Size([32, 9, 768])
epoch #  4      train_loss: 0.020       valid_loss: 0.731                                                                                                                                                   

Stopping early
Start Prediction
0.8045454545454546
             precision    recall  f1-score   support

          0       0.83      0.78      0.80       107
          1       0.90      0.89      0.90       101
          2       0.76      0.77      0.76        99
          3       0.72      0.78      0.75        92
          4       0.93      0.87      0.90       107
          5       0.88      0.75      0.81       117
          6       0.70      0.82      0.76        85
          7       0.69      0.81      0.75        85
          8       0.78      0.72      0.75       108
          9       0.81      0.78      0.79       104
         10       0.85      0.89      0.87        95

   accuracy                           0.80      1100
  macro avg       0.80      0.81      0.80      1100
weighted avg       0.81      0.80      0.81      1100
```
## Domain-specific pretraining
```
python run_pretraining.py   --input_file=tmp/tf_examples.tfrecord   --output_dir=tmp/pretraining_output   --do_train=True   --do_eval=True   --bert_config_file=bert_data/uncased_L-12_H-768_A-12/bert_config.json   --init_checkpoint=bert_data/uncased_L-12_H-768_A-12/bert_model.ckpt   --train_batch_size=16   --max_seq_length=128   --max_predictions_per_seq=20   --num_train_steps=50000   --num_warmup_steps=5000 --learning_rate=2e-5
```

```
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 50000
INFO:tensorflow:  loss = 1.5952029
INFO:tensorflow:  masked_lm_accuracy = 0.6594538
INFO:tensorflow:  masked_lm_loss = 1.553228
INFO:tensorflow:  next_sentence_accuracy = 0.97875
INFO:tensorflow:  next_sentence_loss = 0.04855649
```

```
python run_pretraining.py   --input_file=tmp/tf_examples.tfrecord   --output_dir=tmp/pretraining_output   --do_train=True   --do_eval=True   --bert_config_file=bert_data/uncased_L-12_H-768_A-12/bert_config.json   --init_checkpoint=bert_data/uncased_L-12_H-768_A-12/bert_model.ckpt   --train_batch_size=16   --max_seq_length=128   --max_predictions_per_seq=20   --num_train_steps=100000   --num_warmup_steps=10000 --learning_rate=2e-5
```
```
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 100000
INFO:tensorflow:  loss = 1.3737804
INFO:tensorflow:  masked_lm_accuracy = 0.68698704
INFO:tensorflow:  masked_lm_loss = 1.3770859
INFO:tensorflow:  next_sentence_accuracy = 1.0
INFO:tensorflow:  next_sentence_loss = 0.0028481877
```
