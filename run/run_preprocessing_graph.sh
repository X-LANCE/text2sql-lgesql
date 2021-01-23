#!/bin/bash

train='data/train.bin'
dev='data/dev.bin'
table='data/tables.bin'
train_out='data/train.dgl.bin'
dev_out='data/dev.dgl.bin'

echo "Start to preprocess the original train dataset ..."
python3 -u preprocess/process_dataset_graph.py --dataset_path ${train} --table_path ${table} --output_path ${train_out} #--verbose > train.log
echo "Start to preprocess the original dev dataset ..."
python3 -u preprocess/process_dataset_graph.py --dataset_path ${dev} --table_path ${table} --output_path ${dev_out} #--verbose > dev.log
