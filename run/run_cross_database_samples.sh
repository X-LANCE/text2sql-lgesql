#!/bin/bash

train_data='data/train.json'
train_out='data/train.cross.bin'
table_out='data/tables.bin'

echo "Start to preprocess the original train dataset ..."
python -u preprocess/dataset_process.py --cross_database --dataset_path ${train_data} --processed_table_path ${table_out} --output_path ${train_out}
