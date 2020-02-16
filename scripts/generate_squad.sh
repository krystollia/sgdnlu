export SQUAD_DIR=~/data/squad_orig
export SQUAD_VERSION=v1.1
export BERT_BASE_DIR=~/models/tf2.1/keras_bert/uncased_L-12_H-768_A-12
export OUTPUT_DIR=~/data/squad

python official/nlp/bert/create_finetuning_data.py \
 --squad_data_file=${SQUAD_DIR}/train-${SQUAD_VERSION}.json \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --train_data_output_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_meta_data \
 --fine_tuning_task_type=squad --max_seq_length=384
