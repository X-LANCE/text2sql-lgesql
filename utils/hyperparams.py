#coding=utf8
import sys, os

EXP_PATH = 'exp'

def hyperparam_path(args, task='ratsql'):
    if args.read_model_path and args.testing:
        return args.read_model_path
    if task in ['ratsql', 'ratsql_golden']:
        exp_path = hyperparam_path_ratsql(args)
    elif task == 'ratsql_relation_dropout':
        exp_path = hyperparam_path_relation_dropout(args)
    elif task == 'ratsql_coarse2fine':
        exp_path = hyperparam_path_coarse2fine(args)
    elif task == 'graph_pruning':
        exp_path = hyperparam_path_graph_pruning(args)
    else:
        raise ValueError('Unrecognized task name %s' % (task))
    return os.path.join(EXP_PATH, args.task, exp_path)

def hyperparam_path_relation_dropout(args):
    exp_path = 'dropout_%s__pattern_%s__method_%s' % (args.relation_dropout, args.relation_dropout_pattern, args.relation_dropout_method)
    return exp_path

def hyperparam_path_graph_pruning(args):
    exp_path = ''
    # encoder params
    exp_path += 'emb_%s' % (args.embed_size) if args.ptm is None else 'ptm_%s' % (args.ptm)
    exp_path += '__gnn_%s_x_%s_shared_%s' % (args.gnn_hidden_size, args.gnn_num_layers, args.shared_num_layers)
    exp_path += '__head_%s' % (args.num_heads)
    exp_path += '__dp_%s' % (args.dropout)
    exp_path += '__attndp_%s' % (args.attn_drop)
    exp_path += '__method_%s' % (args.question_pooling_method)
    exp_path += '__score_%s_mlp_%s' % (args.score_function, args.dim_reduction)
    exp_path += '__bce_ls_%s_pos_%s' % (args.label_smoothing, args.pos_weight) if args.loss_function == 'bce' else \
            '__focal_alpha_%s_gamma_%s' % (args.alpha, args.gamma)
    # training params
    exp_path += '__bsize_%s' % (args.batch_size)
    exp_path += '__lr_%s' % (args.lr) if args.ptm is None else '__lr_%s_decay_%s' % (args.lr, args.layerwise_decay)
    exp_path += '__l2_%s' % (args.l2)
    exp_path += '__warmup_%s' % (args.warmup_ratio)
    exp_path += '__schedule_%s' % (args.lr_schedule)
    exp_path += '__me_%s' % (args.max_epoch)
    exp_path += '__mn_%s' % (args.max_norm)
    return exp_path

def hyperparam_path_ratsql(args):
    exp_path = ''
    # encoder params
    exp_path += 'emb_%s' % (args.embed_size) if args.ptm is None else 'ptm_%s' % (args.ptm)
    exp_path += '__gnn_%s_x_%s' % (args.gnn_hidden_size, args.gnn_num_layers)
    exp_path += '__head_%s' % (args.num_heads)
    exp_path += '__dp_%s' % (args.dropout)
    exp_path += '__dpa_%s' % (args.attn_drop)
    exp_path += '__dpc_%s' % (args.drop_connect)
    # decoder params
    exp_path += '__cell_%s_%s_x_%s' % (args.lstm, args.lstm_hidden_size, args.lstm_num_layers)
    exp_path += '_chunk_%s' % (args.chunk_size) if args.lstm == 'onlstm' else ''
    exp_path += '_no' if args.no_parent_state else ''
    exp_path += '__attvec_%s' % (args.att_vec_size)
    exp_path += '__sepcxt' if args.sep_cxt else '__jointcxt'
    exp_path += '_no' if args.no_context_feeding else ''
    exp_path += '__ae_%s' % (args.action_embed_size)
    exp_path += '_no' if args.no_parent_production_embed else ''
    exp_path += '__fe_%s' % ('no' if args.no_parent_field_embed else args.field_embed_size)
    exp_path += '__te_%s' % ('no' if args.no_parent_field_type_embed else args.type_embed_size)
    # training params
    exp_path += '__bsize_%s' % (args.batch_size)
    exp_path += '__lr_%s' % (args.lr) if args.ptm is None else '__lr_%s_decay_%s' % (args.lr, args.layerwise_decay)
    exp_path += '__l2_%s' % (args.l2)
    exp_path += '__warmup_%s' % (args.warmup_ratio)
    exp_path += '__schedule_%s' % (args.lr_schedule)
    exp_path += '__me_%s' % (args.max_epoch)
    exp_path += '__mn_%s' % (args.max_norm)
    exp_path += '__beam_%s' % (args.beam_size)
    return exp_path

def hyperparam_path_coarse2fine(args):
    exp_path = '' if args.ptm is None else 'ptm_%s_subword_%s_schema_%s__' % (args.ptm, args.subword_aggregation, args.schema_aggregation)
    exp_path += 'dp_%s__rdp_%s__' % (args.dropout, args.drop_connect)
    exp_path += 'gnn_%s_x_%s_shared_%s' % (args.gnn_hidden_size, args.gnn_num_layers, args.shared_num_layers)
    # exp_path += '__method_%s' % (args.question_pooling_method)
    # exp_path += '__score_%s_mlp_%s' % (args.score_function, args.dim_reduction)
    # exp_path += '__bce_ls_%s_pos_%s' % (args.label_smoothing, args.pos_weight) if 'bce' in args.loss_function else \
            # '__focal_alpha_%s_gamma_%s' % (args.alpha, args.gamma)
    # exp_path += '_margin_%s' % (args.margin) if 'margin' in args.loss_function else ''
    # exp_path += '_softmax' if 'softmax' in args.loss_function else ''
    exp_path += '__bsize_%s' % (args.batch_size)
    exp_path += '__lr_%s' % (args.lr) if args.ptm is None else '__lr_%s_decay_%s' % (args.lr, args.layerwise_decay)
    exp_path += '__l2_%s' % (args.l2)
    exp_path += '__warmup_%s' % (args.warmup_ratio)
    exp_path += '__schedule_%s' % (args.lr_schedule)
    exp_path += '__me_%s' % (args.max_epoch)
    exp_path += '__mn_%s' % (args.max_norm)
    # exp_path += '__prune_%s_coeffi_%s' % ('yes' if args.prune else 'no', args.prune_coeffi)
    # exp_path += '__sample_min_%s_max_%s' % (args.min_rate, args.max_rate)
    return exp_path
