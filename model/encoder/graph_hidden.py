#coding=utf8
import copy, math
import torch, dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

class LineGraphHiddenLayer(nn.Module):

    def __init__(self, hidden_size, relation_num, khops=4, num_layers=8, num_heads=8, feat_drop=0.2, attn_drop=0.):
        super(LineGraphHiddenLayer, self).__init__()
        self.num_layers = num_layers
        self.relation_embed = nn.Embedding(relation_num, hidden_size)
        self.gnn_layers = nn.ModuleList([LGNNLayer(
            hidden_size, khops=khops, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop
        ) for _ in range(self.num_layers)])
        self.dropout_layer = nn.Dropout(p=feat_drop)

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
        self.linear_self = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_khops = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(khops)])
        self.linear_fuse = nn.Linear(self.hidden_size, self.hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
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
        prev_x = self.linear_self(x)
        khops_x = aggregate_neighbours(self.khops, g, x)
        khops_x = sum([linear(x) for linear, x in zip(self.linear_khops, khops_x)])
        edge_x = self.linear_fuse(torch.mm(pmpd, lg_x))
        outputs = F.gelu(prev_x + khops_x + edge_x)
        # feedforward module
        outputs = self.layernorm1(x + outputs)
        outputs = self.layernorm2(outputs + self.feedforward(outputs))
        return outputs

def aggregate_neighbours(k, g, z):
    # initializing list to collect message passing result
    z_list = []
    with g.local_scope():
        g.ndata['z'] = z
        # pulling message from 1-hop neighbourhood
        g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
        z_list.append(g.ndata['z'])
        for i in range(k - 1):
            # pulling message from k-hop neighborhood
            g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
            z_list.append(g.ndata['z'])
    return z_list
