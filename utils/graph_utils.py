#coding=utf8
import numpy as np
import dgl, torch
from scipy.sparse import coo_matrix, block_diag

def sparse2th(mat):
    value = mat.data
    indices = torch.LongTensor([mat.row, mat.col])
    tensor = torch.sparse.FloatTensor(indices, torch.from_numpy(value).float(), mat.shape)
    return tensor

class GraphExample():

    pass

class BatchedGraph():

    pass

class GraphFactory():

    def __init__(self, method='lgnn', add_cls=True, relation_vocab=None):
        super(GraphFactory, self).__init__()
        self.method = eval('self.' + method)
        self.batch_method = eval('self.batch_' + method)
        self.add_cls, self.relation_vocab = add_cls, relation_vocab

    def graph_construction(self, ex: dict, db: dict):
        """ Wrapper function """
        q = np.array(ex['relations'], dtype='<U100')
        s = np.array(db['relations'], dtype='<U100')
        q_s = np.array(ex['schema_linking'][0], dtype='<U100')
        s_q = np.array(ex['schema_linking'][1], dtype='<U100')
        if self.add_cls:
            cls_cls = np.array(['cls-cls-identity'], dtype='<U100')[np.newaxis, :]
            cls_q = np.array(['cls-question'] * q.shape[0], dtype='<U100')[np.newaxis, :]
            q_cls = np.array(['question-cls'] * q.shape[0], dtype='<U100')[:, np.newaxis]
            cls_s = np.array(['cls-table'] * len(db['table_names']) + ['cls-column'] * len(db['column_names']), dtype='<U100')[np.newaxis, :]
            s_cls = np.array(['table-cls'] * len(db['table_names']) + ['column-cls'] * len(db['column_names']), dtype='<U100')[:, np.newaxis]
            relation = np.concatenate([
                np.concatenate([cls_cls, cls_q, cls_s], axis=1),
                np.concatenate([q_cls, q, q_s], axis=1),
                np.concatenate([s_cls, s_q, s], axis=1)
            ], axis=0)
        else:
            relation = np.concatenate([
                np.concatenate([q, q_s], axis=1),
                np.concatenate([s_q, s], axis=1)
            ], axis=0)
        relation = relation.flatten().tolist()
        return self.method(ex, db, relation)

    def batch_graphs(self, ex_list, device, train=True, **kwargs):
        """ Batch graphs in example list """
        return self.batch_method(ex_list, device, train=train, **kwargs)

    def lgnn(self, ex, db, relation):
        # filter some relations to avoid too many nodes in the line graph
        filter_relations = ['question-question', 'table-table', 'column-column',
            'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable', 'table-column', 'column-table',
            'table-question-nomatch', 'question-table-nomatch', 'column-question-nomatch', 'question-column-nomatch',
            'cls-cls-identity', 'question-question-dist0', 'table-table-identity', 'column-column-identity']
        num_nodes = int(math.sqrt(len(relation)))
        edges = [(idx // num_nodes, idx % num_nodes, self.relation_vocab[r]) for idx, r in enumerate(relation) if r not in filter_relations]
        num_edges = len(edges)
        src_ids, dst_ids = list(map(lambda r: r[0], edges)), list(map(lambda r: r[1], edges))
        rel_ids = list(map(lambda r: r[2], edges))

        graph = GraphExample()
        graph.g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.edge_feat = torch.tensor(rel_ids, dtype=torch.long)
        src_p = coo_matrix(([1.0] * num_edges, (src_ids, range(num_edges))), shape=(num_nodes, num_edges))
        dst_p = coo_matrix(([1.0] * num_edges, (dst_ids, range(num_edges))), shape=(num_nodes, num_edges))
        graph.incidence_matrix = (src_p, dst_p)
        return graph

    def rgat(self, ex, db, relation):
        # alter some relations naming
        relation_mapping_dict = {
            "*-*-identity": 'column-column-identity', "*-question": "column-question-nomatch", "question-*": "question-column-nomatch",
            "*-table": "column-table", "table-*": "table-column", "*-column": "column-column", "column-*": "column-column"
        }
        num_nodes = int(math.sqrt(len(relation)))
        edges = [(idx // num_nodes, idx % num_nodes, self.relation_vocab[
            (relation_mapping_dict[r] if r in relation_mapping_dict else r)])
            for idx, r in enumerate(relation)]
        num_edges = len(edges)
        src_ids, dst_ids = list(map(lambda r: r[0], edges)), list(map(lambda r: r[1], edges))
        rel_ids = list(map(lambda r: r[2], edges))
        
        graph = GraphExample()
        graph.g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.edge_feat = torch.tensor(rel_ids, dtype=torch.long)
        return graph

    def batch_lgnn(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        # g = dgl.batch([ex.g for ex in graph_list]).to(device)
        # bg.g = (g, dgl.khop_graph(g, 2).to(device), dgl.khop_graph(g, 3).to(device), dgl.khop_graph(g, 4).to(device))
        # lg = g.line_graph(backtracking=False)
        # bg.lg = (lg, dgl.khop_graph(lg, 2).to(device), dgl.khop_graph(lg, 3).to(device), dgl.khop_graph(lg, 4).to(device))
        g = dgl.batch([ex.g for ex in graph_list])
        bg.g = (g.add_self_loop().to(device), dgl.khop_graph(g, 2).add_self_loop().to(device),
            dgl.khop_graph(g, 3).add_self_loop().to(device), dgl.khop_graph(g, 4).add_self_loop().to(device))
        lg = g.line_graph(backtracking=False)
        bg.lg = (lg.add_self_loop().to(device), dgl.khop_graph(lg, 2).add_self_loop().to(device),
            dgl.khop_graph(lg, 3).add_self_loop().to(device), dgl.khop_graph(lg, 4).add_self_loop().to(device))
        bg.edge_feat = torch.cat([ex.edge_feat for ex in graph_list], dim=0).to(device)
        src_p = sparse2th(block_diag([ex.incidence_matrix[0] for ex in graph_list])).to(device)
        dst_p = sparse2th(block_diag([ex.incidence_matrix[1] for ex in graph_list])).to(device)
        bg.incidence_matrix = (src_p, dst_p)
        return bg

    def batch_rgat(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        g = dgl.batch([ex.g for ex in graph_list])
        bg.g = g.to(device)
        bg.edge_feat = torch.cat([ex.edge_feat for ex in graph_list], dim=0).to(device)
        return bg

    def rgcn(self, entry, db):
        pass