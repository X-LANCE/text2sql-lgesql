task=lgnn_plus_rat
seed=$1
device=0
read_model_path='saved_models/electra-large'

python scripts/hetgnn_checkpoint.py --task $task --seed $seed --device $device --read_model_path $read_model_path
