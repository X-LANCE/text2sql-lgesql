#coding=utf8
import copy, math
import torch, dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GATConv
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable

@Registrable.register('hetgnn')
class LGNN(nn.Module):

    def __init__(self, args):
        super(LGNN, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.relation_embed = nn.Embedding(args.relation_num, args.hidden_size)
        self.gnn_layers = nn.ModuleList([LGNNLayer(
            args.hidden_size, khops=args.khops, num_heads=args.num_heads, feat_drop=args.dropout, attn_drop=args.attn_drop
        ) for _ in range(self.num_layers)])
        self.dropout_layer = nn.Dropout(p=args.dropout)

    def forward(self, x, batch):
        """
            x: num_nodes x hidden_size
            batch.graph.g and batch.graph.lg: dgl graph and its line graph
            batch.graph.incidence_matrix: tuple of num_nodes x num_edges sparse float matrix, src and dst connections
        """
        # prepare inputs
        lg_x = self.relation_embed(batch.graph.edge_feat)
        g, lg = batch.graph.g, batch.graph.lg
        pm, pd = batch.graph.incidence_matrix
        pmpd = pm + pd
        lg_pmpd = torch.transpose(pmpd, 0, 1)
        # iteration
        for i in range(self.num_layers):
            x, lg_x = self.dropout_layer(x), self.dropout_layer(lg_x)
            x, lg_x = self.gnn_layers[i](g, lg, x, lg_x, pmpd, lg_pmpd)
        return x, lg_x

class LGNNLayer(nn.Module):

    def __init__(self, hidden_size, khops=4, num_heads=8, feat_drop=0.2, attn_drop=0.):
        super(LGNNLayer, self).__init__()
        self.node_update_layer = LGNNCore(hidden_size, khops, num_heads, feat_drop, attn_drop)
        self.edge_update_layer = LGNNCore(hidden_size, khops, num_heads, feat_drop, attn_drop)

    def forward(self, g, lg, x, lg_x, pmpd, lg_pmpd):
        next_x = self.node_update_layer(g, x, lg_x, pmpd)
        next_lg_x = self.edge_update_layer(lg, lg_x, x, lg_pmpd)
        return next_x, next_lg_x

class LGNNCore(nn.Module):

    def __init__(self, hidden_size, khops=4, num_heads=8, feat_drop=0.2, attn_drop=0.):
        super(LGNNCore, self).__init__()
        self.hidden_size = hidden_size
        self.khops = khops
        # self.linear_khops = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size // khops) for _ in range(khops)])
        self.gat_khops = nn.ModuleList([GATConv(self.hidden_size, self.hidden_size // khops, 1) for _ in range(khops)])
        self.linear_fuse = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size *4, self.hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(self.hidden_size) # intermediate module
        self.layernorm2 = nn.LayerNorm(self.hidden_size) # after feedforward module
        self.dropout_layer = nn.Dropout(p=feat_drop)
        self.attndrop_layer = nn.Dropout(p=attn_drop)

    def forward(self, g, x, lg_x, pmpd):
        """ @Params:
            g: dgl.graph obj
            x: node feats
            lg_x: edge feats
            pmpd: incidence matrix, sparse FloatTensor, num_nodes x num_edges
        """
        # khops_x = aggregate_neighbours(self.khops, g, x)
        # khops_x = [linear(x) for linear, x in zip(self.linear_khops, khops_x)]
        khops_x = [gat(tmp_g, x).squeeze(1) for gat, tmp_g in zip(self.gat_khops, g)]
        outputs = torch.cat(khops_x, dim=-1)
        edge_x = torch.mm(pmpd, lg_x)
        fuse_x = self.linear_fuse(torch.cat([outputs, edge_x], dim=-1))
        outputs = self.layernorm1(x + fuse_x)
        # feedforward module
        outputs = self.layernorm2(outputs + self.feedforward(outputs))
        return outputs

def aggregate_neighbours(k, g, z):
    # initializing list to collect message passing result
    z_list = []
    for i in range(k):
        tmp_g = g[i]
        with tmp_g.local_scope():
            tmp_g.ndata['z'] = z
            tmp_g.update_all(fn.copy_src(src='z', out='m'), fn.mean(msg='m', out='z'))
            z_list.append(tmp_g.ndata['z'])
    # with g.local_scope():
        # g.ndata['z'] = z
        # pulling message from 1-hop neighbourhood
        # g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
        # z_list.append(g.ndata['z'])
        # for i in range(k - 1):
            # pulling message from k-hop neighborhood
            # g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
            # z_list.append(g.ndata['z'])
    return z_list
       