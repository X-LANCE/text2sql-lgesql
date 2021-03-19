#coding=utf8
import copy, math
import torch, dgl
import torch.nn as nn
from model.model_utils import Registrable
from model.encoder.rat import RATLayer
from model.encoder.lgnn import EdgeUpdateLayerMetaPath, EdgeUpdateLayerNodeAttention, EdgeUpdateLayerNodeAffine

@Registrable.register('lgnn_concat_rat')
class LGNNConcatRAT(nn.Module):
    """ Compared with RAT, we utilize a line graph to explicitly model the propagation among edges:
    1. aggregate info from nearby edges via GCN/GAT
    2. aggregate info from src and dst nodes
    """
    def __init__(self, args):
        super(LGNNConcatRAT, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.relation_num = args.relation_num
        self.relation_share_layers = args.relation_share_layers
        self.relation_share_heads = args.relation_share_heads
        self.ndim = args.gnn_hidden_size # node feature dim
        # we pay more attention to node feats, thus could share edge feats in multi-heads to reduce dimension
        assert args.num_heads >= 2, 'Num of heads should be at least 2'
        self.edim = args.gnn_hidden_size // args.num_heads if self.relation_share_heads else args.gnn_hidden_size // 2
        if self.relation_share_layers:
            self.relation_embed = nn.Embedding(self.relation_num, self.edim) # contain global relations
        else:
            self.relation_embed = nn.ModuleList([nn.Embedding(self.relation_num, self.edim) for _ in range(self.num_layers)])
        self.gnn_layers = nn.ModuleList([LGNNConcatRATLayer(self.ndim, self.edim, num_heads=args.num_heads, feat_drop=args.dropout)
            for _ in range(self.num_layers)])

    def forward(self, x, batch):
        if self.relation_share_layers:
            lg_x = self.relation_embed(batch.graph.edge_feat)
            lg_x_full = self.relation_embed(batch.graph.full_edge_feat)
        else:
            lg_x = self.relation_embed[0](batch.graph.edge_feat)
        src_ids, dst_ids = batch.graph.g.edges(order='eid')
        for i in range(self.num_layers):
            # lg_x_full is directly extracted in each layer
            # lg_x = self.relation_embed[i](batch.graph.edge_feat) if not self.relation_share_layers else lg_x
            lg_x_full = self.relation_embed[i](batch.graph.full_edge_feat) if not self.relation_share_layers else lg_x_full
            x, lg_x = self.gnn_layers[i](x, lg_x_full, lg_x, batch.graph.full_g, batch.graph.g, batch.graph.lg, src_ids.long(), dst_ids.long())
        return x, lg_x

class LGNNConcatRATLayer(nn.Module):

    def __init__(self, ndim, edim, num_heads=8, feat_drop=0.2):
        super(LGNNConcatRATLayer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        self.node_update = NodeUpdateLayer(self.ndim, self.edim, self.num_heads, feat_drop)
        self.edge_update = EdgeUpdateLayerMetaPath(self.edim, self.ndim, self.num_heads, feat_drop=feat_drop)

    def forward(self, x, lg_x_full, lg_x, full_g, g, lg, src_ids, dst_ids):
        """ Different strategies to update nodes and edges:
        1. parallel scheme
        2. first update node, then use new node feats to update edge
        3. first update edge, then use new edge feats to update node
        """
        # parallel
        out_x, _ = self.node_update(x, lg_x_full, lg_x, full_g, g)
        # return out_x, lg_x
        src_x = torch.index_select(x, dim=0, index=src_ids)
        dst_x = torch.index_select(x, dim=0, index=dst_ids)
        out_lg_x, _ = self.edge_update(lg_x, src_x, dst_x, lg)

        # node update first
        # out_x, _ = self.node_update(x, lg_x_full, lg_x, full_g, g)
        # src_x = torch.index_select(out_x, dim=0, index=src_ids)
        # dst_x = torch.index_select(out_x, dim=0, index=dst_ids)
        # out_lg_x, _ = self.edge_update(lg_x, src_x, dst_x, lg)

        # edge update first
        # src_x = torch.index_select(x, dim=0, index=src_ids)
        # dst_x = torch.index_select(x, dim=0, index=dst_ids)
        # out_lg_x, _ = self.edge_update(lg_x, src_x, dst_x, lg)
        # out_x, _ = self.node_update(x, lg_x_full, out_lg_x, full_g, g)
        return out_x, out_lg_x

class NodeUpdateLayer(RATLayer):

    def forward(self, x, lg_x_full, lg_x, full_g, g):
        """ @Params:
                x: node feats, num_nodes x ndim
                lg_x_full: all edge feats, num_edges x edim
                lg_x: local edge feats, num_edges x edim
                full_g: dgl.graph, a complete graph for node update
                g: dgl.graph, a local graph for node update
        """
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        q, k, v = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k), v.view(-1, self.num_heads, self.d_k)
        with full_g.local_scope():
            full_g.ndata['q'], full_g.ndata['k'] = q[:, :self.num_heads // 2], k[:, :self.num_heads // 2]
            full_g.ndata['v'] = v[:, :self.num_heads // 2]
            full_g.edata['e'] = lg_x_full.view(-1, self.num_heads // 2, self.d_k) if lg_x_full.size(-1) == self.d_k * self.num_heads // 2 else \
                lg_x_full.unsqueeze(1).expand(-1, self.num_heads // 2, -1)
            out_x_1 = self.propagate_attention(full_g)
        with g.local_scope():
            g.ndata['q'], g.ndata['k'] = q[:, self.num_heads // 2:], k[:, self.num_heads // 2:]
            g.ndata['v'] = v[:, self.num_heads // 2:]
            g.edata['e'] = lg_x.view(-1, self.num_heads // 2, self.d_k) if lg_x.size(-1) == self.d_k * self.num_heads // 2 else \
                lg_x.unsqueeze(1).expand(-1, self.num_heads // 2, -1)
            out_x_2 = self.propagate_attention(g)
        out_x = torch.cat([out_x_1, out_x_2], dim=1)
        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.ndim)))
        out_x = self.ffn(out_x)
        return out_x, lg_x
