# NLI_BERT
not paragraph
max_seq_length = 512 train_batch=4 epoch=24 base cased
eval_accuracy = 0.6587807, eval_loss = 3.3459177, global_step = 59394, loss = 3.345302
eval_accuracy = 0.7042766, eval_loss = 3.2103376, global_step = 59394, loss = 3.231809

 python run_classifier.py \
   --task_name=MRPC \
   --do_predict=true \
   --data_dir=$GLUE_DIR/MRPC \
   --vocab_file=/vocab.txt \
   --bert_config_file=/bert_config.json \
   --init_checkpoint=$TRAINED_CLASSIFIER \
   --max_seq_length=512 \
   --output_dir=/tmp/


python my_run.py   --task_name=TOEFL   --do_train=true   --do_eval=true   --data_dir=TOEFL11   --vocab_file=cased_L-12_H-768_A-12/vocab.txt   --bert_config_file=cased_L-12_H-768_A-12/bert_config.json   --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt   --max_seq_length=512   --train_batch_size=8   --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=tmp/ --do_lower_case=False


python my_run.py   --task_name=TOEFL   --do_predict=true   --data_dir=TOEFL11   --vocab_file=cased_L-12_H-768_A-12/vocab.txt   --bert_config_file=cased_L-12_H-768_A-12/bert_config.json   --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt   --max_seq_length=512   --output_dir=tmp/  --do_lower_case=False


python my_run_paragraph.py   --task_name=TOEFL   --do_train=true   --do_eval=true   --data_dir=TOEFL11   --vocab_file=cased_L-12_H-768_A-12/vocab.txt   --bert_config_file=cased_L-12_H-768_A-12/bert_config.json   --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt   --max_seq_length=220   --train_batch_size=16   --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=tmp2/ --do_lower_case=False

python my_run_paragraph.py   --task_name=TOEFL   --do_predict=true   --data_dir=TOEFL11   --vocab_file=cased_L-12_H-768_A-12/vocab.txt   --bert_config_file=cased_L-12_H-768_A-12/bert_config.json   --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt   --max_seq_length=220   --output_dir=tmp_paragraph/  --do_lower_case=False
