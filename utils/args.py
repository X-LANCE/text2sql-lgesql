#coding=utf-8
import argparse
import sys

def init_args(params=sys.argv[1:], task='ratsql'):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    arg_parser = add_argument_text2sql(arg_parser)
    if task in ['ratsql_coarse2fine', 'graph_pruning']:
        arg_parser = add_argument_graph_pruning(arg_parser)
        arg_parser = add_argument_coarse2fine(arg_parser)
    if task == 'ratsql_relation_dropout':
        arg_parser = add_argument_relation_dropout(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt

def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--task', default='text2sql', help='task name')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--preprocess', action='store_true', help='whether read examples from preprocessed dataset')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--read_model_path', type=str, help='read pretrained model path')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
    arg_parser.add_argument('--grad_accumulate', default=1, type=int, help='accumulate grad and update once every x steps')
    arg_parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    arg_parser.add_argument('--layerwise_decay', type=float, default=1.0, help='layerwise decay rate for lr, used for ptm')
    arg_parser.add_argument('--l2', type=float, default=1e-4, help='weight decay coefficient')
    arg_parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup steps proportion')
    arg_parser.add_argument('--lr_schedule', default='linear', choices=['constant', 'linear', 'ratsql', 'cosine'], help='lr scheduler')
    arg_parser.add_argument('--eval_after_epoch', default=40, type=int, help='Start to evaluate after x epoch')
    arg_parser.add_argument('--load_optimizer', action='store_true', default=False, help='Whether to load optimizer state')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    arg_parser.add_argument('--max_norm', default=5., type=float, help='clip gradients')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--ptm', type=str, choices=['bert-base-uncased', 'bert-large-uncased', 'bert-large-uncased-whole-word-masking', 'roberta-base', 'roberta-large', 'electra-base-discriminator', 'electra-large-discriminator'], help='pretrained model name')
    arg_parser.add_argument('--subword_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling'], default='mean', help='aggregate subword from PTM')
    arg_parser.add_argument('--schema_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling', 'head+tail'], default='head+tail', help='aggregate schema repr')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--gnn_num_layers', default=8, type=int, help='num of GNN layers in encoder')
    arg_parser.add_argument('--gnn_hidden_size', default=256, type=int, help='Size of GNN layers hidden states')
    arg_parser.add_argument('--num_heads', default=8, type=int, help='num of heads in multihead attn')
    arg_parser.add_argument('--attn_drop', type=float, default=0., help='attn dropout rate in GAT encoding module')
    return arg_parser

def add_argument_text2sql(arg_parser):
    # decoder hyperparams
    arg_parser.add_argument('--lstm', choices=['lstm', 'onlstm'], default='onlstm', help='Type of LSTM used, ONLSTM or traditional LSTM')
    arg_parser.add_argument('--chunk_size', default=8, type=int, help='parameter of ONLSTM')
    arg_parser.add_argument('--att_vec_size', default=512, type=int, help='size of attentional vector')
    arg_parser.add_argument('--sep_cxt', action='store_true', help='when calculating context vectors, use seperate cxt for question and schema')
    arg_parser.add_argument('--drop_connect', type=float, default=0.2, help='recurrent connection dropout rate in decoder lstm')
    arg_parser.add_argument('--lstm_num_layers', type=int, default=1, help='num_layers of decoder')
    arg_parser.add_argument('--lstm_hidden_size', default=512, type=int, help='Size of LSTM hidden states')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='Embedding size of ASDL fields')
    arg_parser.add_argument('--type_embed_size', default=64, type=int, help='Embeddings ASDL types')
    arg_parser.add_argument('--no_context_feeding', action='store_true', default=False,
                            help='Do not use embedding of context vectors')
    arg_parser.add_argument('--no_parent_production_embed', default=False, action='store_true',
                            help='Do not use embedding of parent ASDL production to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_embed', default=False, action='store_true',
                            help='Do not use embedding of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_type_embed', default=False, action='store_true',
                            help='Do not use embedding of the ASDL type of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_state', default=False, action='store_true',
                            help='Do not use the parent hidden state to update decoder LSTM state')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_step', default=100, type=int, help='Maximum number of time steps used in decoding')
    return arg_parser

def add_argument_coarse2fine(arg_parser):
    arg_parser.add_argument('--prune', action='store_true', help='whether pruning the full schema graph for second pass encoding or decoding')
    arg_parser.add_argument('--prune_coeffi', type=float, default=1.0, help='coefficient for pruning loss')
    arg_parser.add_argument('--shared_num_layers', type=int, default=8, help='the bottom layers are shared for pruning and end2end encoding')
    arg_parser.add_argument('--min_rate', type=float, default=.05, help='minimum sampling rate for irrelevant schema items during text2sql decoding')
    arg_parser.add_argument('--max_rate', type=float, default=1., help='maximum sampling rate for irrelevant schema items during text2sql decoding')
    return arg_parser

def add_argument_graph_pruning(arg_parser):
    arg_parser.add_argument('--question_pooling_method', default='multihead-attention', choices=['max-pooling', 'mean-pooling', 'attentive-pooling', 'multihead-attention'], help='method to aggregate the representation for the question sequence')
    arg_parser.add_argument('--score_function', default='biaffine', choices=['dot', 'bilinear', 'affine', 'biaffine'], help='score function to calculate similarity score given two vectors')
    arg_parser.add_argument('--dim_reduction', default=4, type=int, help='perform dim reduction before score function')
    arg_parser.add_argument('--loss_function', default='bce', choices=['bce', 'focal'], help='binary loss function used to calculate loss')
    arg_parser.add_argument('--label_smoothing', default=0.15, type=float, help='label smoothing factor used for binary cross entropy loss, 0 =< ls < 0.5')
    arg_parser.add_argument('--pos_weight', default=1.0, type=float, help='weight for positive labels')
    arg_parser.add_argument('--alpha', default=0.8, type=float, help='parameter for data imbalance')
    arg_parser.add_argument('--gamma', default=0.5, type=float, help='parameter for kaiming focal loss function')
    return arg_parser

def add_argument_relation_dropout(arg_parser):
    arg_parser.add_argument('--relation_dropout', type=float, default=0.2, help='relation dropout rate, as data augmentation')
    arg_parser.add_argument('--relation_dropout_pattern', default='cross', choices=['all', 'cross'], help='drop which area in the adjacency matrix: all area or only the schema linking rectangle')
    arg_parser.add_argument('--relation_dropout_method', default='linear', choices=['constant', 'linear'], help='how to change the dropout ratio')
    return arg_parser
