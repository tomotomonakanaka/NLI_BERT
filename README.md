# NLI_MODELS
## Setup
```
$ unzip TOEFL11.zip
$ pip install pytorch # depending on your device
$ pip install dgl # depending on your device
$ pip install -r requirements.txt
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
learning_rate = 5e-5
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
