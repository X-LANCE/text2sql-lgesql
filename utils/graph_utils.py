#coding=utf8
import numpy as np
import dgl, torch
from utils.example import Example
import scipy as sc
from scipy.sparse import coo_matrix

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

    def __init__(self, method='lgnn'):
        super(GraphFactory, self).__init__()
        self.method = eval('self.' + method)
        self.batch_method = eval('self.batch_' + method)

    def graph_construction(self, ex: dict, db: dict):
        """ Wrapper function """
        return self.method(ex, db)

    def batch_graphs(self, ex_list, device, train=True, **kwargs):
        """ Batch graphs in example list """
        return self.batch_method(ex_list, device, train=train, **kwargs)

    def lgnn(self, ex, db):
        q = np.array(ex['relations'], dtype='<U100')
        s = np.array(db['relations'], dtype='<U100')
        q_s = np.array(ex['schema_linking'][0], dtype='<U100')
        s_q = np.array(ex['schema_linking'][1], dtype='<U100')
        num_nodes = len(ex['processed_question_toks']) + len(db['table_names']) + len(db['column_names'])
        if Example.add_cls:
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
            num_nodes += 1
        else:
            relation = np.concatenate([
                np.concatenate([q, q_s], axis=1),
                np.concatenate([s_q, s], axis=1)
            ], axis=0)

        relation = relation.flatten().tolist()
        # filter some relations to avoid too many nodes in the line graph
        filter_relations = ['question-question', 'table-table', 'column-column',
            'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable', 'table-column', 'column-table',
            'table-question-nomatch', 'question-table-nomatch', 'column-question-nomatch', 'question-column-nomatch',
            'cls-cls-identity', 'question-question-dist0', 'table-table-identity', 'column-column-identity']
        edges = [(idx // num_nodes, idx % num_nodes, Example.relation_vocab[r]) for idx, r in enumerate(relation) if r not in filter_relations]
        num_edges = len(edges)
        src_ids, dst_ids = list(map(lambda r: r[0], edges)), list(map(lambda r: r[1], edges))
        rel_ids = list(map(lambda r: r[2], edges))
        
        graph = GraphExample()
        graph.g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.edge_feat = torch.tensor(rel_ids, dtype=torch.long).unsqueeze(1)
        src_p = coo_matrix(([1.0] * num_edges, (src_ids, range(num_edges))), shape=(num_nodes, num_edges))
        dst_p = coo_matrix(([1.0] * num_edges, (dst_ids, range(num_edges))), shape=(num_nodes, num_edges))
        graph.incidence_matrix = (src_p, dst_p)
        return graph
    
    def batch_lgnn(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.g = dgl.batch([ex.g for ex in graph_list]).to(device)
        bg.lg = bg.g.line_graph(backtracking=False)
        bg.edge_feat = torch.cat([ex.edge_feat for ex in graph_list], dim=0).to(device)
        src_p = sparse2th(sc.block_diag([ex.incidence_matrix[0] for ex in graph_list])).to(device)
        dst_p = sparse2th(sc.block_diag([ex.incidence_matrix[1] for ex in graph_list])).to(device)
        bg.incidence_matrix = (src_p, dst_p)
        return bg

    def rgcn(self, entry, db):
        pass

    def rgat(self, entry, db):
        pass