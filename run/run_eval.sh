#!/bin/bash
task=eval_model
device=0
#read_model_path='exp/ratsql/rat__qtc__ptm_electra-base-discriminator__gnn_512_x_8__rel_share__head_8__rel_share__dp_0.2__dpa_0.0__dpc_0.2__bsize_20__lr_0.0002_decay_0.8__l2_0.1__warmup_0.1__schedule_linear__me_100__mn_5.0__beam_5'
read_model_path='exp/lgnn/lgnn_concat_rat_gp_node_0.15__ptm_electra-base-discriminator__gnn_512_x_8__rel_share__head_8__rel_share__dp_0.2__dpa_0.0__dpc_0.2__bsize_20__lr_0.0002_decay_0.8__l2_0.1__warmup_0.1__schedule_linear__me_100__mn_5.0__beam_5'

python scripts/hetgnn.py --task $task --testing --device $device --read_model_path $read_model_path
