task=ratsql_golden_$1
seed=999
device=0
preprocess='--preprocess'
testing='' #'--testing'
read_model_path='' #"--read_model_path exp/ratsql_golden_remove_col0_$1/emb_300__gnn_256_x_8__head_8__dp_0.2__attndp_0.0__cell_onlstm_512_x_1_chunk_8__attvec_512__jointcxt_no__ae_128__fe_64__te_64__init_0.2__bsize_20__lr_0.0005__l2_0.0001__warmup_0.1__schedule_linear__me_100__mn_5.0__beam_5/"

ptm='' # '--ptm bert-base-uncased'
embed_size=300
gnn_hidden_size=256
gnn_num_layers=8
num_heads=8
dropout=0.2
attn_drop=0.0
drop_connect=0.0

lstm=onlstm
chunk_size=8
att_vec_size=512
sep_cxt=''
lstm_hidden_size=512
lstm_num_layers=1
action_embed_size=128
field_embed_size=64
type_embed_size=64
no_context_feeding='--no_context_feeding'
no_parent_production_embed=''
no_parent_field_embed=''
no_parent_field_type_embed=''
no_parent_state=''

decode_max_step=100
batch_size=20
grad_accumulate=1
lr=5e-4
l2=1e-4
warmup_ratio=0.1
lr_schedule=linear
eval_after_epoch=40
max_epoch=100
max_norm=5
beam_size=5

python scripts/ratsql_golden.py --task $task --seed $seed --device $device $preprocess $testing $read_model_path \
    $ptm --gnn_hidden_size $gnn_hidden_size --dropout $dropout --attn_drop $attn_drop --att_vec_size $att_vec_size \
    --embed_size $embed_size --gnn_num_layers $gnn_num_layers --num_heads $num_heads $sep_cxt \
    --lstm $lstm --chunk_size $chunk_size --drop_connect $drop_connect --lstm_hidden_size $lstm_hidden_size --lstm_num_layers $lstm_num_layers --decode_max_step $decode_max_step \
    --action_embed_size $action_embed_size --field_embed_size $field_embed_size --type_embed_size $type_embed_size \
    $no_context_feeding $no_parent_production_embed $no_parent_field_embed $no_parent_field_type_embed $no_parent_state \
    --batch_size $batch_size --grad_accumulate $grad_accumulate --lr $lr --l2 $l2 --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule --eval_after_epoch $eval_after_epoch \
    --max_epoch $max_epoch --max_norm $max_norm --beam_size $beam_size   
