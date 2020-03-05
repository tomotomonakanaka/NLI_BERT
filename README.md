# NLI_MODELS
## Setup
```
$ unzip TOEFL_sentence.zip
$ pip install pytorch # depending on your device
$ pip install -r requirements.txt
$ mkdir save_model
```
### run simple BERT model ###
```
$ python simple_BERT.py
```

```
epoch #  1	train_loss: 1.769	valid_loss: 1.211
epoch #  2	train_loss: 0.935	valid_loss: 0.843
epoch #  3	train_loss: 0.529	valid_loss: 0.991
epoch #  4	train_loss: 0.270	valid_loss: 1.084
epoch #  5	train_loss: 0.113	valid_loss: 1.196

              precision    recall  f1-score   support

           0       0.69      0.79      0.74       100
           1       0.86      0.86      0.86       100
           2       0.79      0.77      0.78       100
           3       0.69      0.67      0.68       100
           4       0.75      0.85      0.80       100
           5       0.75      0.77      0.76       100
           6       0.72      0.71      0.72       100
           7       0.72      0.73      0.72       100
           8       0.80      0.70      0.74       100
           9       0.76      0.71      0.74       100
          10       0.79      0.76      0.78       100

    accuracy                           0.76      1100
   macro avg       0.76      0.76      0.76      1100
weighted avg       0.76      0.76      0.76      1100
```
## run BERT model
```
$ python BERT_paragraph.py
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
8. resizing paragraph model for not 128
```

## Paragraph Splitting BERT model
```
$ python BERT_paragraph.py
```

#### Back Ground
```
Total data size: 160434
Max long: 128
batch_size = 32
max_epochs = 4
learning_rate = 6e-5
              precision    recall  f1-score   support

           0       0.36      0.39      0.37      1188
           1       0.47      0.53      0.50      1766
           2       0.46      0.40      0.43      1653
           3       0.43      0.46      0.45      1814
           4       0.50      0.45      0.47      1260
           5       0.45      0.42      0.44      1689
           6       0.42      0.38      0.40      1882
           7       0.36      0.36      0.36      1339
           8       0.50      0.47      0.49      1702
           9       0.34      0.40      0.37      1702
          10       0.44      0.45      0.44      1763

    accuracy                           0.43     17758
   macro avg       0.43      0.43      0.43     17758
weighted avg       0.43      0.43      0.43     17758

              precision    recall  f1-score   support

           0       0.82      0.83      0.82        99
           1       0.95      0.90      0.92       106
           2       0.78      0.87      0.82        90
           3       0.87      0.69      0.77       127
           4       0.85      0.93      0.89        91
           5       0.84      0.90      0.87        93
           6       0.81      0.81      0.81       100
           7       0.82      0.75      0.78       110
           8       0.72      0.86      0.78        84
           9       0.86      0.85      0.86       101
          10       0.90      0.91      0.90        99

    accuracy                           0.84      1100
   macro avg       0.84      0.84      0.84      1100
weighted avg       0.84      0.84      0.84      1100

0.8381818181818181
```

```
Start Prediction
              precision    recall  f1-score   support

           0       0.34      0.36      0.35      1188
           1       0.48      0.54      0.51      1766
           2       0.46      0.42      0.44      1653
           3       0.44      0.44      0.44      1814
           4       0.52      0.46      0.49      1260
           5       0.43      0.43      0.43      1689
           6       0.44      0.41      0.43      1882
           7       0.37      0.36      0.37      1339
           8       0.51      0.48      0.50      1702
           9       0.36      0.41      0.38      1702
          10       0.44      0.47      0.46      1763

    accuracy                           0.44     17758
   macro avg       0.44      0.43      0.43     17758
weighted avg       0.44      0.44      0.44     17758

              precision    recall  f1-score   support

           0       0.83      0.84      0.83        99
           1       0.94      0.91      0.93       103
           2       0.84      0.84      0.84       100
           3       0.84      0.72      0.77       117
           4       0.82      0.92      0.87        89
           5       0.84      0.90      0.87        93
           6       0.84      0.84      0.84       100
           7       0.85      0.81      0.83       105
           8       0.76      0.86      0.81        88
           9       0.83      0.81      0.82       103
          10       0.90      0.87      0.89       103

    accuracy                           0.84      1100
   macro avg       0.84      0.85      0.84      1100
weighted avg       0.85      0.84      0.84      1100

0.8445454545454546
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
