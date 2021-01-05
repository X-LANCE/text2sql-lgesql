#coding=utf8
import torch
import torch.nn as nn
from model.encoder.graph_input import *
from model.encoder.graph_hidden import *
from model.encoder.graph_output import *
from model.model_utils import Registrable

@Registrable.register('encoder_ratsql')
class RATSQLGraphEncoder(nn.Module):

    def __init__(self, args):
        super(RATSQLGraphEncoder, self).__init__()
        lazy_load = args.lazy_load if hasattr(args, 'lazy_load') else False
        self.input_layer = RATSQLGraphInputLayer(args.embed_size, args.gnn_hidden_size, args.word_vocab, dropout=args.dropout) \
            if args.ptm is None else RATSQLGraphInputLayerPTM(args.ptm, args.gnn_hidden_size, dropout=args.dropout,
                subword_aggregation=args.subword_aggregation, schema_aggregation=args.schema_aggregation, lazy_load=lazy_load)
        self.hidden_layer = RATSQLGraphHiddenLayer(args.gnn_hidden_size, args.relation_num, args.num_heads,
            num_layers=args.gnn_num_layers, feat_drop=args.dropout, attn_drop=args.attn_drop,
            share_layers=True, share_heads=True, layerwise_relation=False)
        self.output_layer = RATSQLGraphOutputLayer(args.gnn_hidden_size)

    def forward(self, batch):
        outputs = self.input_layer(batch)
        outputs = self.hidden_layer(outputs, batch.relations, batch.relations_mask)
        encodings, mask = self.output_layer(outputs, batch)
        return encodings, mask

@Registrable.register('encoder_coarse2fine')
class Coarse2FineGraphEncoder(nn.Module):

    def __init__(self, args):
        super(Coarse2FineGraphEncoder, self).__init__()
        lazy_load = args.lazy_load if hasattr(args, 'lazy_load') else False
        self.input_layer = RATSQLGraphInputLayer(args.embed_size, args.gnn_hidden_size, args.word_vocab, dropout=args.dropout) \
            if args.ptm is None else RATSQLGraphInputLayerPTM(args.ptm, args.gnn_hidden_size, dropout=args.dropout,
                subword_aggregation=args.subword_aggregation, schema_aggregation=args.schema_aggregation, lazy_load=lazy_load)
        self.shared_num_layers = args.shared_num_layers
        assert 0 <= args.shared_num_layers <= args.gnn_num_layers
        if self.shared_num_layers > 0:
            self.hidden_layer = RATSQLGraphHiddenLayer(args.gnn_hidden_size, args.relation_num, args.num_heads,
                num_layers=self.shared_num_layers, feat_drop=args.dropout, attn_drop=args.attn_drop,
                share_layers=True, share_heads=True, layerwise_relation=False)
        self.output_layer = Coarse2FineGraphOutputLayer(args)
        if args.gnn_num_layers > args.shared_num_layers:
            # share relational embedding modules (k and v) for both shared and two private gnn encoding modules
            if args.shared_num_layers > 0:
                self.output_layer.private_sql_encoder.relation_embed_k = self.hidden_layer.relation_embed_k
                self.output_layer.private_sql_encoder.relation_embed_v = self.hidden_layer.relation_embed_v
            self.output_layer.private_aux_encoder.relation_embed_k = self.output_layer.private_sql_encoder.relation_embed_k
            self.output_layer.private_aux_encoder.relation_embed_v = self.output_layer.private_sql_encoder.relation_embed_v

    def forward(self, batch, mode='text2sql'):
        outputs = self.input_layer(batch)
        if self.shared_num_layers > 0:
            outputs = self.hidden_layer(outputs, batch.relations, batch.relations_mask)
        return self.output_layer(outputs, batch, mode)
