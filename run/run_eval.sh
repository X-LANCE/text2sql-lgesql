#!/bin/bash
task=evaluation
device=0
read_model_path=''

python scripts/text2sql.py --task $task --testing --device $device --read_model_path $read_model_path