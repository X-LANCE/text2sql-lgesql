#coding=utf8
import torch
import torch.nn as nn
from model.encoder.graph_input import *
# from model.encoder.rgat import RGAT
# from model.encoder.lgnn import LGNN
from model.encoder.rat import RAT
from model.encoder.graph_output import *
from model.model_utils import Registrable

@Registrable.register('encoder_hetgnn')
class HetGNNEncoder(nn.Module):

    def __init__(self, args):
        super(HetGNNEncoder, self).__init__()
        lazy_load = args.lazy_load if hasattr(args, 'lazy_load') else False
        self.input_layer = GraphInputLayer(args.embed_size, args.gnn_hidden_size, args.word_vocab, dropout=args.dropout,
            schema_aggregation=args.schema_aggregation, add_cls=args.add_cls) \
            if args.ptm is None else GraphInputLayerPTM(args.ptm, args.gnn_hidden_size, dropout=args.dropout, add_cls=args.add_cls,
                subword_aggregation=args.subword_aggregation, schema_aggregation=args.schema_aggregation, lazy_load=lazy_load)
        self.hidden_layer = Registrable.by_name(args.model)(args)
        self.output_layer = GraphOutputLayer(args.gnn_hidden_size)

    def forward(self, batch):
        outputs = self.input_layer(batch)
        outputs, _ = self.hidden_layer(outputs, batch)
        encodings, mask = self.output_layer(outputs, batch)
        return encodings, mask