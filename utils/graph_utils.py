#coding=utf8
import numpy as np
import dgl, torch, math
from scipy.sparse import coo_matrix, block_diag
import time

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
        # we only allow edge flow to *, no edge start with *
        filter_relations = [
            'question-question', 'table-table', 'column-column', 'table-column', 'column-table',
            'question-question-dist-2', 'question-question-dist2'
            'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable',
            '*-column', 'column-*',
            # '*-table', 'table-*',
            # 'table-question-nomatch', 'question-table-nomatch', 'column-question-nomatch', 'question-column-nomatch', 'question-*', '*-question',
            'cls-cls-identity', 'question-question-dist0', 'table-table-identity', 'column-column-identity', '*-*-identity'
        ]
        # filter some relations to avoid too many nodes in the line graph
        relation_mapping_dict = {
            'question-*': 'question-column-nomatch',
            '*-question': 'column-question-nomatch',
            'table-*': 'table-column-has',
            '*-table': 'column-table-has'
        }
        num_nodes = int(math.sqrt(len(relation)))
        edges = [(idx // num_nodes, idx % num_nodes, (relation_mapping_dict[r] if r in relation_mapping_dict else r))
            for idx, r in enumerate(relation) if r not in filter_relations]
        num_edges = len(edges)
        src_ids, dst_ids = list(map(lambda r: r[0], edges)), list(map(lambda r: r[1], edges))
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], edges))

        graph = GraphExample()
        graph.g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.edge_feat = torch.tensor(rel_ids, dtype=torch.long)
        # construct line graph, remove some edges in the line graph
        # graph.lg = graph.g.line_graph(backtracking=False).remove_self_loop().add_self_loop()
        lg = graph.g.line_graph(backtracking=False)
        match_ids = [idx for idx, r in enumerate(edges) if 'match' in r[2]]
        src, dst, eids = lg.edges(form='all', order='eid')
        eids = [e for u, v, e in zip(src.tolist(), dst.tolist(), eids.tolist()) if not (u in match_ids and v in match_ids)]
        graph.lg = lg.edge_subgraph(eids, preserve_nodes=True).remove_self_loop().add_self_loop()
        # print(graph.g.num_nodes(), graph.g.num_edges(), graph.lg.num_nodes(), graph.lg.num_edges())
        # if not (not (graph.g.in_degrees() == 0).any().item() and not (graph.lg.in_degrees() == 0).any()):
            # print(ex['question'], ex['query'])
        return graph

    def rat(self, ex, db, relation):
        filter_relations = [
            # 'question-question', 'table-table', 'column-column', 'table-column', 'column-table',
            # 'question-question-dist-2', 'question-question-dist2'
            # 'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable',
            # '*-table', '*-column', 'table-*', 'column-*',
            # 'question-*', '*-question',
            # 'table-question-nomatch', 'question-table-nomatch', 'column-question-nomatch', 'question-column-nomatch',
            # 'cls-cls-identity', 'question-question-dist0', 'table-table-identity', 'column-column-identity', '*-*-identity'
        ]
        relation_mapping_dict = {
            "*-*-identity": 'column-column-identity', "*-question": "column-question-nomatch", "question-*": "question-column-nomatch",
            "*-table": "column-table", "table-*": "table-column", "*-column": "column-column", "column-*": "column-column"
        }
        num_nodes = int(math.sqrt(len(relation)))
        edges = [(idx // num_nodes, idx % num_nodes, (relation_mapping_dict[r] if r in relation_mapping_dict else r))
            for idx, r in enumerate(relation) if r not in filter_relations]
        num_edges = len(edges)
        src_ids, dst_ids = list(map(lambda r: r[0], edges)), list(map(lambda r: r[1], edges))
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], edges))

        graph = GraphExample()
        graph.g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.edge_feat = torch.tensor(rel_ids, dtype=torch.long)
        return graph

    def batch_lgnn(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.g = dgl.batch([ex.g for ex in graph_list]).to(device)
        bg.lg = dgl.batch([ex.lg for ex in graph_list]).to(device)
        bg.edge_feat = torch.cat([ex.edge_feat for ex in graph_list], dim=0).to(device)
        return bg

    def batch_rat(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.g = dgl.batch([ex.g for ex in graph_list]).to(device)
        bg.edge_feat = torch.cat([ex.edge_feat for ex in graph_list], dim=0).to(device)
        return bg
