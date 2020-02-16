export GLUE_DIR=~/data/glue_data
export BERT_BASE_DIR=~/models/tf2.1/keras_bert/uncased_L-12_H-768_A-12

export TASK_NAME=MRPC
export OUTPUT_DIR=~/data/glue
python official/nlp/bert/create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME}
