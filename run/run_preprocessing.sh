#!/bin/bash

train_data='data/train.json'
dev_data='data/dev.json'
table_data='data/tables.json'
train_out='data/train.stanza.bin'
dev_out='data/dev.stanza.bin'
table_out='data/tables.stanza.bin'
glove_embedding_file='data/.cache/glove.42B.300d.txt'
vocab_glove='data/vocab_glove.txt'
vocab='data/vocab.stanza.txt'

echo "Start to preprocess the original train dataset ..."
#python3 preprocess/dataset_process.py --dataset_path ${train_data} --processed_table_path ${table_out} --output_path ${train_out} #--verbose > train.log
python3 -u preprocess/dataset_process.py --dataset_path ${train_data} --raw_table_path ${table_data} --processed_table_path ${table_out} --output_path ${train_out} #--verbose > train.log
echo "Start to preprocess the original dev dataset ..."
python3 -u preprocess/dataset_process.py --dataset_path ${dev_data} --processed_table_path ${table_out} --output_path ${dev_out} #--verbose > dev.log
echo "Start to build word vocab for the dataset ..."
awk -v FS=' ' '{print $1}' ${glove_embedding_file} > ${vocab_glove}
python3 -u preprocess/build_vocab.py --data_paths ${train_out} --table_path ${table_out} --reference_file ${vocab_glove} --mwf 4 --output_path ${vocab}
