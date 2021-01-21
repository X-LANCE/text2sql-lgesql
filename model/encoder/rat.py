#coding=utf8
import copy, math
import torch, dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable

@Registrable.register('rat')
class RAT(nn.Module):

    def __init__(self, args):
        super(RAT, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.relation_num = args.relation_num
        self.relation_share_layers, self.relation_share_heads = args.relation_share_layers, args.relation_share_heads
        edim = args.gnn_hidden_size // args.num_heads if self.relation_share_heads else args.gnn_hidden_size
        if self.relation_share_layers:
            self.relation_embed = nn.Embedding(args.relation_num, edim)
        else:
            self.relation_embed = nn.ModuleList([nn.Embedding(args.relation_num, edim) for _ in range(self.num_layers)])
        self.gnn_layers = nn.ModuleList([RATLayer(args.gnn_hidden_size, num_heads=args.num_heads, feat_drop=args.dropout)
            for _ in range(self.num_layers)])

    def forward(self, x, batch):
        if self.relation_share_layers:
            lg_x = self.relation_embed(batch.graph.edge_feat)
        for i in range(self.num_layers):
            lg_x = self.relation_embed[i](batch.graph.edge_feat) if not self.relation_share_layers else lg_x
            x, lg_x = self.gnn_layers[i](x, lg_x, batch.graph.g)
        return x, lg_x

class RATLayer(nn.Module):

    def __init__(self, hidden_size, num_heads=8, feat_drop=0.2):
        super(RATLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = self.hidden_size // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.hidden_size, self.hidden_size),\
            nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.affine_o = nn.Linear(self.hidden_size, self.hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(self.hidden_size)
        self.layernorm2 = nn.LayerNorm(self.hidden_size)
        self.feat_dropout = nn.Dropout(p=feat_drop)

    def forward(self, x, lg_x, g):
        """ @Params:
                x: node feats, num_nodes x hidden_size
                lg_x: edge feats, num_edges x hidden_size
                g: dgl.graph
        """
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        e = lg_x.unsqueeze(1).expand(-1, self.num_heads, -1) if lg_x.size(-1) == self.d_k else \
            lg_x.view(-1, self.num_heads, self.d_k) # not relation_share_heads
        e = self.feat_dropout(e)
        with g.local_scope():
            g.ndata['q'], g.ndata['k'] = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k)
            g.ndata['v'] = v.view(-1, self.num_heads, self.d_k)
            g.edata['e'] = e
            out_x = self.propagate_attention(g)

        out_x = self.layernorm1(x + self.affine_o(out_x.view(-1, self.hidden_size)))
        out_x = self.layernorm2(out_x + self.feedforward(out_x))
        return out_x, lg_x

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_sum_edge_dot_dst('k', 'q', 'e', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(src_sum_edge_dot_edge('v', 'e', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x

def src_sum_edge_dot_dst(src_field, dst_field, e_field, out_field):
    def func(edges):
        return {out_field: ((edges.src[src_field] + edges.data[e_field]) * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}

    return func

def src_sum_edge_dot_edge(src_field, e_field1, e_field2, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] + edges.data[e_field1]) * edges.data[e_field2]}

    return func

def div_by_z(in_field, norm_field, out_field):
    def func(nodes):
        return {out_field: nodes.data[in_field] / nodes.data[norm_field]}

    return func
