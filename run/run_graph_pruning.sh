task=graph_pruning_pretrain
seed=999
device=0
preprocess='--preprocess'
testing='' #'--testing'
read_model_path='' #'--read_model_path exp/graph_pruning/..'

ptm='bert-base-uncased'
embed_size=300
gnn_hidden_size=256
gnn_num_layers=8
num_heads=8
dropout=0.2
attn_drop=0.0
drop_connect=0.2

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
decode_max_step=100

question_pooling_method=multihead-attention # attentive-pooling, multihead-attention, pointwise-attentive-pooling
score_function=fnn # fnn, dot, bilinear
label_smoothing=0.15
loss_function=bce
prune='--prune' # '--prune'
shared_num_layers=8 # 0 <= shared_num_layers <= gnn_num_layers
prune_coeffi=1.0

python scripts/graph_pruning.py --task $task --seed $seed --device $device $preprocess $testing $read_model_path \
    $ptm --dropout $dropout --attn_drop $attn_drop --drop_connect $drop_connect \
    --embed_size $embed_size --gnn_hidden_size $gnn_hidden_size --gnn_num_layers $gnn_num_layers --num_heads $num_heads \
    --lstm $lstm --chunk_size $chunk_size --lstm_hidden_size $lstm_hidden_size --lstm_num_layers $lstm_num_layers --att_vec_size $att_vec_size \
    --action_embed_size $action_embed_size --field_embed_size $field_embed_size --type_embed_size $type_embed_size \
    $sep_cxt $no_context_feeding $no_parent_production_embed $no_parent_field_embed $no_parent_field_type_embed $no_parent_state \
    --batch_size $batch_size --grad_accumulate $grad_accumulate --lr $lr --l2 $l2 --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule --eval_after_epoch $eval_after_epoch \
    --max_epoch $max_epoch --max_norm $max_norm --beam_size $beam_size --decode_max_step $decode_max_step \
    --question_pooling_method $question_pooling_method --score_function $score_function --label_smoothing $label_smoothing --loss_function $loss_function \
    $prune --prune_coeffi $prune_coeffi --shared_num_layers $shared_num_layers
