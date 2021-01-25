#coding=utf8
import copy, math
import torch, dgl
import torch.nn as nn
from model.model_utils import Registrable
from model.encoder.rat import RATLayer as NodeUpdateLayer
from model.encoder.lgnn import EdgeUpdateLayerMetaPath, EdgeUpdateLayerNodeAttention, EdgeUpdateLayerNodeAffine

@Registrable.register('lgnn_plus_rat')
class LGNNPlusRAT(nn.Module):
    """ Compared with RAT, we utilize a line graph to explicitly model the propagation among edges:
    1. aggregate info from nearby edges via GCN/GAT
    2. aggregate info from src and dst nodes
    """
    def __init__(self, args):
        super(LGNNPlusRAT, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.relation_num = args.relation_num
        self.relation_share_heads = args.relation_share_heads
        self.ndim = args.gnn_hidden_size # node feature dim
        # we pay more attention to node feats, thus could share edge feats in multi-heads to reduce dimension
        self.edim = args.gnn_hidden_size // args.num_heads if self.relation_share_heads else args.gnn_hidden_size
        self.relation_embed = nn.Embedding(self.relation_num, self.edim) # contain global relations
        self.gnn_layers = nn.ModuleList([LGNNPlusRATLayer(self.ndim, self.edim, num_heads=args.num_heads, feat_drop=args.dropout)
            for _ in range(self.num_layers)])

    def forward(self, x, batch):
        lg_x = self.relation_embed(batch.graph.edge_feat)
        lg_x_local = lg_x[batch.graph.local_index]
        src_ids, dst_ids = batch.graph.src_ids, batch.graph.dst_ids
        for i in range(self.num_layers):
            x, lg_x, lg_x_local = self.gnn_layers[i](x, lg_x, lg_x_local, batch.graph.g, batch.graph.lg, src_ids, dst_ids, batch.graph.local_index)
        return x, lg_x_local

class LGNNPlusRATLayer(nn.Module):

    def __init__(self, ndim, edim, num_heads=8, feat_drop=0.2):
        super(LGNNPlusRATLayer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        self.node_update = NodeUpdateLayer(self.ndim, self.edim, self.num_heads, feat_drop)
        self.edge_update = EdgeUpdateLayerMetaPath(self.edim, self.ndim, self.num_heads, feat_drop=feat_drop)

    def forward(self, x, lg_x, lg_x_local, g, lg, src_ids, dst_ids, local_index):
        """ Different strategies to update nodes and edges:
        1. parallel scheme
        2. first update node, then use new node feats to update edge
        3. first update edge, then use new edge feats to update node
        """
        # parallel
        out_x, _ = self.node_update(x, lg_x, g)
        src_x = torch.index_select(x, dim=0, index=src_ids)
        dst_x = torch.index_select(x, dim=0, index=dst_ids)
        out_lg_x_local, _ = self.edge_update(lg_x_local, src_x, dst_x, lg)
        out_lg_x = lg_x.masked_scatter_(local_index.unsqueeze(-1), out_lg_x_local)

        # node update first
        # out_x, _ = self.node_update(x, lg_x, g)
        # src_x = torch.index_select(out_x, dim=0, index=src_ids)
        # dst_x = torch.index_select(out_x, dim=0, index=dst_ids)
        # out_lg_x_local, _ = self.edge_update(lg_x_local, src_x, dst_x, lg)
        # out_lg_x = lg_x.masked_scatter_(local_index.unsqueeze(-1), out_lg_x_local)

        # edge update first
        # src_x = torch.index_select(x, dim=0, index=src_ids)
        # dst_x = torch.index_select(x, dim=0, index=dst_ids)
        # out_lg_x_local, _ = self.edge_update(lg_x_local, src_x, dst_x, lg)
        # out_lg_x = lg_x.masked_scatter_(local_index.unsqueeze(-1), out_lg_x_local)
        # out_x, _ = self.node_update(x, out_lg_x, g)
        return out_x, out_lg_x, out_lg_x_local
