#coding=utf8
import copy, math
import torch, dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable
from model.encoder.rat import RATLayer as NodeUpdateLayer
from model.encoder.rat import scaled_exp, div_by_z
from model.encoder.graph_output import MultiHeadAttention

@Registrable.register('lgnn')
class LGNN(nn.Module):
    """ Compared with RAT, we utilize a line graph to explicitly model the propagation among edges:
    1. aggregate info from nearby edges via GCN/GAT
    2. aggregate info from src and dst nodes
    """
    def __init__(self, args):
        super(LGNN, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.relation_num = args.relation_num
        self.relation_share_heads = args.relation_share_heads
        self.ndim = args.gnn_hidden_size # node feature dim
        # we pay more attention to node feats, thus share edge feats in multi-heads to reduce dimension
        self.edim = args.gnn_hidden_size // args.num_heads if self.relation_share_heads else args.gnn_hidden_size
        self.relation_embed = nn.Embedding(self.relation_num, self.edim)
        self.gnn_layers = nn.ModuleList([LGNNLayer(self.ndim, self.edim, num_heads=args.num_heads, feat_drop=args.dropout)
            for _ in range(self.num_layers)])

    def forward(self, x, batch):
        lg_x = self.relation_embed(batch.graph.edge_feat)
        src_ids, dst_ids = batch.graph.g.edges(order='eid')
        for i in range(self.num_layers):
            x, lg_x = self.gnn_layers[i](x, lg_x, batch.graph.g, batch.graph.lg, src_ids.long(), dst_ids.long())
        return x, lg_x

class LGNNLayer(nn.Module):

    def __init__(self, ndim, edim, num_heads=8, feat_drop=0.2):
        super(LGNNLayer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        self.node_update = NodeUpdateLayer(self.ndim, self.num_heads, feat_drop)
        self.edge_update = EdgeUpdateLayerNodeAffine(self.edim, self.ndim, feat_drop)

    def forward(self, x, lg_x, g, lg, src_ids, dst_ids):
        """ Different strategies to update nodes and edges:
        1. parallel scheme
        2. first update node, then use new node feats to update edge
        3. first update edge, then use new edge feats to update node
        """
        # parallel
        out_x, _ = self.node_update(x, lg_x, g)
        src_x = torch.index_select(x, dim=0, index=src_ids)
        dst_x = torch.index_select(x, dim=0, index=dst_ids)
        out_lg_x, _ = self.edge_update(lg_x, src_x, dst_x, lg)

        # node update first
        # out_x, _ = self.node_update(x, lg_x, g)
        # src_x = torch.index_select(out_x, dim=0, index=src_ids)
        # dst_x = torch.index_select(out_x, dim=0, index=dst_ids)
        # out_lg_x, _ = self.edge_update(lg_x, src_x, dst_x, lg)

        # edge update first
        # src_x = torch.index_select(x, dim=0, index=src_ids)
        # dst_x = torch.index_select(x, dim=0, index=dst_ids)
        # out_lg_x, _ = self.edge_update(lg_x, src_x, dst_x lg)
        # out_x, _ = self.node_update(x, out_lg_x, g)
        return out_x, out_lg_x

class EdgeUpdateLayerNodeAffine(nn.Module):

    def __init__(self, edim, ndim, feat_drop=0.2):
        super(EdgeUpdateLayerNodeAffine, self).__init__()
        self.edim, self.ndim = edim, ndim
        self.affine = nn.Linear(self.edim + self.ndim * 2, self.edim)
        self.feedforward = nn.Sequential(
            nn.Linear(self.edim, self.edim * 4),
            nn.ReLU(),
            nn.Linear(self.edim * 4, self.edim)
        )
        self.layernorm1 = nn.LayerNorm(self.edim)
        self.layernorm2 = nn.LayerNorm(self.edim)
        self.feat_dropout = nn.Dropout(p=feat_drop)

    def forward(self, x, src_x, dst_x, g):
        concat_x = self.affine(self.feat_dropout(torch.cat([x, src_x, dst_x], dim=-1)))
        out_x = torch.tanh(concat_x)
        out_x = self.layernorm1(x + out_x)
        out_x = self.layernorm2(out_x + self.feedforward(out_x))
        return out_x, (src_x, dst_x)

class EdgeUpdateLayerNodeAttention(nn.Module):

    def __init__(self, edim, ndim, num_heads=8, feat_drop=0.2):
        super(EdgeUpdateLayerNodeAttention, self).__init__()
        self.edim, self.ndim = edim, ndim
        self.multihead_attn = MultiHeadAttention(self.ndim, self.edim, self.ndim, dropout=feat_drop, heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(self.edim, self.edim * 4),
            nn.ReLU(),
            nn.Linear(self.edim * 4, self.edim)
        )
        self.layernorm1 = nn.LayerNorm(self.edim)
        self.layernorm2 = nn.LayerNorm(self.edim)

    def forward(self, x, src_x, dst_x, g):
        out_x = self.multihead_attn(torch.stack([src_x, dst_x], dim=1), x.unsqueeze(1)).squeeze(1)
        out_x = self.layernorm1(x + out_x)
        out_x = self.layernorm2(out_x + self.feedforward(out_x))
        return out_x, (src_x, dst_x)

class EdgeUpdateLayerMetapath(nn.Module):

    def __init__(self, edim, ndim, num_heads=8, feat_drop=0.2):
        super(EdgeUpdateLayerMetapath, self).__init__()
        self.edim, self.ndim = edim, ndim
        self.num_heads = num_heads
        self.d_k = self.ndim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.edim, self.ndim), \
            nn.Linear(self.edim, self.ndim), nn.Linear(self.ndim, self.ndim)
        self.affine_o = nn.Linear(self.ndim, self.edim)
        self.feedforward = nn.Sequential(
            nn.Linear(self.edim, self.edim * 4),
            nn.ReLU(),
            nn.Linear(self.edim * 4, self.edim)
        )
        self.layernorm1 = nn.LayerNorm(self.edim)
        self.layernorm2 = nn.LayerNorm(self.edim)
        self.feat_dropout = nn.Dropout(p=feat_drop)

    def forward(self, x, src_x, dst_x, g):
        # we do not use node feats src_x and dst_x
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        with g.local_scope():
            g.ndata['q'], g.ndata['k'], g.ndata['v'] = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k), \
                v.view(-1, self.num_heads, self.d_k)
            out_x = self.propagate_attention(g)
        out_x = self.layernorm1(x + self.affine_o(out_x.view(-1, self.ndim)))
        out_x = self.layernorm2(out_x + self.feedforward(out_x))
        return out_x, (src_x, dst_x)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x

class EdgeUpdateLayer(nn.Module):

    def __init__(self, edim, ndim, num_heads=8, feat_drop=0.2):
        super(EdgeUpdateLayerMetapath, self).__init__()
        self.edim, self.ndim = edim, ndim
        self.num_heads = num_heads
        self.d_k = self.ndim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.edim, self.ndim), \
            nn.Linear(self.edim, self.ndim, bias=False), nn.Linear(self.ndim, self.ndim, bias=False)
        self.affine_n = nn.Linear(self.ndim, self.ndim, bias=False)
        self.affine_o = nn.Linear(self.ndim, self.edim)
        self.feedforward = nn.Sequential(
            nn.Linear(self.edim, self.edim * 4),
            nn.ReLU(),
            nn.Linear(self.edim * 4, self.edim)
        )
        self.layernorm1 = nn.LayerNorm(self.edim)
        self.layernorm2 = nn.LayerNorm(self.edim)
        self.feat_dropout = nn.Dropout(p=feat_drop)

    def forward(self, x, src_x, dst_x, g):
        # we do not use node feats src_x and dst_x
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        e = self.affine_n(self.feat_drop(src_x))
        with g.local_scope():
            g.ndata['q'], g.ndata['k'], g.ndata['v'] = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k), \
                v.view(-1, self.num_heads, self.d_k)
            out_x = self.propagate_attention(g)
        out_x = self.layernorm1(x + self.affine_o(out_x.view(-1, self.ndim)))
        out_x = self.layernorm2(out_x + self.feedforward(out_x))
        return out_x, (src_x, dst_x)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func
