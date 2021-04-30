#!/bin/bash
task=evaluation
read_model_path=$1
batch_size=20
beam_size=10
device=-1

python scripts/text2sql.py --task $task --testing --read_model_path $read_model_path \
    --batch_size $batch_size --beam_size $beam_size --device $device
