#coding=utf8
import numpy as np
import dgl, torch, math

class GraphExample():

    pass

class BatchedGraph():

    pass

class GraphFactory():

    def __init__(self, method='rgatsql', relation_vocab=None):
        super(GraphFactory, self).__init__()
        self.method = eval('self.' + method)
        self.batch_method = eval('self.batch_' + method)
        self.relation_vocab = relation_vocab

    def graph_construction(self, ex: dict, db: dict):
        return self.method(ex, db)

    def rgatsql(self, ex, db):
        graph = GraphExample()
        global_edges = ex['graph'].global_edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], global_edges))
        graph.global_edges = torch.tensor(rel_ids, dtype=torch.long)
        graph.global_g = ex['graph'].global_g
        graph.gp = ex['graph'].gp
        graph.question_mask = torch.tensor(ex['graph'].question_mask, dtype=torch.bool)
        graph.schema_mask = torch.tensor(ex['graph'].schema_mask, dtype=torch.bool)
        graph.node_label = torch.tensor(ex['graph'].node_label, dtype=torch.float)
        return graph

    def fast_lgnn_plus_rat(self, ex, db):
        pass
        # graph = self.fast_lgnn(ex, db)
        # # update node graph to a complete graph
        # graph.full_g = ex['graph'].full_g
        # full_edge_feat = list(map(lambda r: self.relation_vocab[r[2]], ex['graph'].full_edges))
        # graph.full_edge_feat = torch.tensor(full_edge_feat, dtype=torch.long)
        # # add extra field local index to extract and scatter local relation feats
        # local_enum, global_enum = graph.edge_feat.size(0), graph.full_edge_feat.size(0)
        # graph.local_index = torch.tensor([1] * local_enum + [0] * (global_enum - local_enum), dtype=torch.bool)
        # return graph

    def fast_lgnn_concat_rat(self, ex, db):
        pass
        # graph = self.fast_lgnn(ex, db)
        # # add a complete graph for nodes
        # full_edge_feat = list(map(lambda r: self.relation_vocab[r[2]], ex['graph'].full_edges))
        # graph.full_edge_feat = torch.tensor(full_edge_feat, dtype=torch.long)
        # graph.full_g = ex['graph'].full_g
        # return graph

    def fast_lgnn(self, ex, db):
        pass
        # graph = GraphExample()
        # edges = ex['graph'].edges
        # rel_ids = list(map(lambda r: self.relation_vocab[r[2]], edges))
        # graph.edge_feat = torch.tensor(rel_ids, dtype=torch.long)
        # graph.g, graph.lg = ex['graph'].g, ex['graph'].lg
        # graph.gp_ng, graph.gp_eg = ex['graph'].gp_ng, ex['graph'].gp_eg
        # graph.context_index = torch.tensor(ex['graph'].context_index, dtype=torch.bool)
        # graph.node_index = torch.tensor(ex['graph'].node_index, dtype=torch.bool)
        # graph.edge_index = torch.tensor(ex['graph'].edge_index, dtype=torch.bool)
        # graph.node_label = torch.tensor(ex['graph'].node_label, dtype=torch.float)
        # graph.edge_label = torch.tensor(ex['graph'].edge_label, dtype=torch.float)
        # return graph

    def batch_graphs(self, ex_list, device, train=True, **kwargs):
        """ Batch graphs in example list """
        return self.batch_method(ex_list, device, train=train, **kwargs)

    def batch_lgnn_plus_rat(self, ex_list, device, train=True, **kwargs):
        bg = self.batch_lgnn(ex_list, device, train=train, **kwargs)
        src_ids, dst_ids = bg.g.edges(order='eid')
        bg.src_ids, bg.dst_ids = src_ids.long(), dst_ids.long()
        bg.g = dgl.batch([ex.graph.full_g for ex in ex_list]).to(device)
        bg.edge_feat = torch.cat([ex.graph.full_edge_feat for ex in ex_list], dim=0).to(device)
        bg.local_index = torch.cat([ex.graph.local_index for ex in ex_list], dim=0).to(device)
        return bg

    def batch_lgnn_concat_rat(self, ex_list, device, train=True, **kwargs):
        bg = self.batch_lgnn(ex_list, device, train=train, **kwargs)
        bg.full_g = dgl.batch([ex.graph.full_g for ex in ex_list]).to(device)
        bg.full_edge_feat = torch.cat([ex.graph.full_edge_feat for ex in ex_list], dim=0).to(device)
        return bg

    def batch_lgnn(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.g = dgl.batch([ex.g for ex in graph_list]).to(device)
        bg.lg = dgl.batch([ex.lg for ex in graph_list]).to(device)
        bg.edge_feat = torch.cat([ex.edge_feat for ex in graph_list], dim=0).to(device)
        bg.context_index = torch.cat([ex.context_index for ex in graph_list], dim=0).to(device)
        bg.node_index = torch.cat([ex.node_index for ex in graph_list], dim=0).to(device)
        smoothing = kwargs.pop('smoothing', 0.0)
        node_label = torch.cat([ex.node_label for ex in graph_list], dim=0)
        node_label = node_label.masked_fill_(~ node_label.bool(), 2 * smoothing) - smoothing
        bg.node_label = node_label.to(device)
        bg.edge_index = torch.cat([ex.edge_index for ex in graph_list], dim=0).to(device)
        edge_label = torch.cat([ex.edge_label for ex in graph_list], dim=0)
        edge_label = edge_label.masked_fill_(~ edge_label.bool(), 2 * smoothing) - smoothing
        bg.edge_label = edge_label.to(device)
        bg.gp_ng = dgl.batch([ex.gp_ng for ex in graph_list]).to(device)
        bg.gp_eg = dgl.batch([ex.gp_eg for ex in graph_list]).to(device)
        return bg

    def batch_rgatsql(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.global_g = dgl.batch([ex.global_g for ex in graph_list]).to(device)
        bg.global_edges = torch.cat([ex.global_edges for ex in graph_list], dim=0).to(device)
        bg.question_mask = torch.cat([ex.question_mask for ex in graph_list], dim=0).to(device)
        bg.schema_mask = torch.cat([ex.schema_mask for ex in graph_list], dim=0).to(device)
        smoothing = kwargs.pop('smoothing', 0.0)
        node_label = torch.cat([ex.node_label for ex in graph_list], dim=0)
        node_label = node_label.masked_fill_(~ node_label.bool(), 2 * smoothing) - smoothing
        bg.node_label = node_label.to(device)
        bg.gp = dgl.batch([ex.gp_ng for ex in graph_list]).to(device)
        return bg