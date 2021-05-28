#!/bin/bash
task=evaluation
read_model_path=saved_models/grappa-mmc-share-73.69
batch_size=20
beam_size=5
device=-1

python scripts/text2sql.py --task $task --testing --read_model_path $read_model_path \
    --batch_size $batch_size --beam_size $beam_size --device $device
