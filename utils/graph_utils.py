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
        self.fast_method = eval('self.fast_' + method)
        self.batch_method = eval('self.batch_' + method)
        self.add_cls, self.relation_vocab = add_cls, relation_vocab

    def graph_construction(self, ex: dict, db: dict, fast: bool = False):
        if fast:
            return self.fast_method(ex, db)
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

    def fast_lgnn_plus_rat(self, ex, db):
        graph = self.fast_lgnn(ex, db)
        full_edge_feat = list(map(lambda r: self.relation_vocab[r[2]], ex['graph'].full_edges))
        graph.full_edge_feat = torch.tensor(full_edge_feat, dtype=torch.long)
        graph.full_g = ex['graph'].full_g
        local_enum, global_enum = graph.edge_feat.size(0), graph.full_edge_feat.size(0)
        graph.local_index = torch.tensor([1] * local_enum + [0] * global_enum, dtype=torch.bool)
        return graph

    def fast_lgnn(self, ex, db):
        graph = GraphExample()
        edges = ex['graph'].edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], edges))
        graph.edge_feat = torch.tensor(rel_ids, dtype=torch.long)
        graph.g, graph.lg = ex['graph'].g, ex['graph'].lg
        graph.gp_ng, graph.gp_eg = ex['graph'].gp_ng, ex['graph'].gp_eg
        graph.context_index = torch.tensor(ex['graph'].context_index, dtype=torch.bool)
        graph.node_index = torch.tensor(ex['graph'].node_index, dtype=torch.bool)
        graph.edge_index = torch.tensor(ex['graph'].edge_index, dtype=torch.bool)
        graph.node_label = torch.tensor(ex['graph'].node_label, dtype=torch.float)
        graph.edge_label = torch.tensor(ex['graph'].edge_label, dtype=torch.float)
        return graph

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

        # graph pruning for nodes
        q_num = len(ex['processed_question_toks']) + 1 if self.add_cls else len(ex['processed_question_toks'])
        s_num = num_nodes - q_num
        graph.context_index = torch.tensor([1] * q_num + [0] * s_num, dtype=torch.bool)
        graph.node_index = ~ graph.context_index
        graph.gp_ng = dgl.heterograph({
            ('context', 'to', 'node'): (list(range(q_num)) * s_num,
            [i for i in range(s_num) for _ in range(q_num)])
            }, num_nodes_dict={'context': q_num, 'node': s_num}, idtype=torch.int32
        )
        t_num = len(db['processed_table_toks'])
        def check_node(i):
            if i < t_num and i in ex['used_tables']:
                return 1.0
            elif i >= t_num and i - t_num in ex['used_columns']:
                return 1.0
            else: return 0.0
        graph.node_label = torch.tensor(list(map(check_node, range(s_num))), dtype=torch.float)

        # graph pruning for edges
        graph.edge_index = torch.tensor(list(map(lambda e: 1 if e[0] >= q_num and e[1] >= q_num else 0, edges)), dtype=torch.bool)
        e_num = graph.edge_index.int().sum().item()
        graph.gp_eg = dgl.heterograph({
            ('context', 'to', 'node'): (list(range(q_num)) * e_num,
            [i for i in range(e_num) for _ in range(q_num)])
            }, num_nodes_dict={'context': q_num, 'node': e_num}, idtype=torch.int32
        )
        def check_edge(t):
            if check_node(t[0]) + check_node(t[1]) > 1.5:
                return 1.0
            else: return 0.0
        graph.edge_label = torch.tensor(list(map(check_edge, filter(lambda e: e[0] >= q_num and e[1] >= q_num, edges))), dtype=torch.float)
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
            "*-table": "column-table-has", "table-*": "table-column-has", "*-column": "column-column", "column-*": "column-column"
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

    def batch_graphs(self, ex_list, device, train=True, **kwargs):
        """ Batch graphs in example list """
        return self.batch_method(ex_list, device, train=train, **kwargs)

    def batch_lgnn_plus_rat(self, ex_list, device, train=True, **kwargs):
        bg = self.batch_lgnn(ex_list, device, train=True, **kwargs)
        bg.full_edge_feat = torch.cat([ex.graph.full_edge_feat for ex in ex_list], dim=0).to(device)
        bg.full_g = dgl.batch([ex.graph.full_g for ex in ex_list]).to(device)
        bg.local_index = torch.cat([ex.graph.local_index for ex in ex_list], dim=0).to(device)
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

    def batch_rat(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.g = dgl.batch([ex.g for ex in graph_list]).to(device)
        bg.edge_feat = torch.cat([ex.edge_feat for ex in graph_list], dim=0).to(device)
        return bg
