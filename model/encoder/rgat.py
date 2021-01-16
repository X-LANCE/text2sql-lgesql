#coding=utf8
import copy, math
import torch, dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable

@Registrable.register('rgat')
class RGAT(nn.Module):

    def __init__(self, args):
        super(RGAT, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.relation_share_layers, self.relation_share_heads = args.relation_share_layers, args.relation_share_heads
        self.relation_embed = nn.Embedding(args.relation_num, args.hidden_size)
        self.gnn_layers = nn.ModuleList([RGATLayer(
            args.hidden_size, num_heads=args.num_heads, feat_drop=args.dropout, attn_drop=args.attn_drop
        ) for _ in range(self.num_layers)])
        self.dropout_layer = nn.Dropout(p=args.dropout)

    def forward(self, x, batch):
        for i in range(self.num_layers):
            x, lg_x = self.dropout_layer(x), self.dropout_layer(lg_x)
            x, lg_x = self.gnn_layers[i](x, lg_x)
        return x, lg_x

class RGATLayer(nn.Module):

    def __init__(self, hidden_size, relation_num, num_heads=8, feat_drop=0.2, attn_drop=0.):
        super(RGATLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = self.hidden_size // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.hidden_size, self.hidden_size),\
            nn.Linear(self.hidden_size, self.hidden_size), nn.Linear(self.hidden_size, self.hidden_size)
        self.affine_o = nn.Linear(self.hidden_size, self.hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(self.hidden_size)
        self.layernorm2 = nn.LayerNorm(self.hidden_size)
        self.attn_dropout = nn.Dropout(p=attn_drop)

    def forward(self, x, lg_x, g, lg):
        """ @Params:
                x: node feats, num_nodes x hidden_size
                lg_x: edge feats, num_edges x hidden_size
        """
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(x), self.affine_k(x), self.affine_v(x)
        with g.local_scope():
            g.ndata['q'], g.ndata['k'] = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k)
            g.edata['r'] = lg_x.unsqueeze(1) if lg_x.size(-1) != self.hidden_size else lg_x # allow for multi-head
            g.app
        
        out_x = self.layernorm1(x + self.affine_o(x))
        out_x = self.layernorm2(out_x + self.feedforward(out_x))
        return out_x, lg_x

def src_dot_dst_plus_edge(src_field, dst_field, edge_field, out_field):
    def func(edges):
        return {out_field: edges.src[src_field] * (edges.dst[dst_field] + edges.data[edge_field]).sum(dim=-1, keepdim=True)}

    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func
