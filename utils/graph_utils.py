#coding=utf8
import numpy as np
import dgl

class GraphProcessor():

    def __init__(self, method='lgnn'):
        super(GraphProcessor, self).__init__()
        self.method = eval('self.' + method)

    def graph_construction(self, entry: dict, db: dict, verbose: bool = False):
        """ Wrapper function """
        return self.method(entry, db, verbose=verbose)

    def lgnn(self, entry, db, verbose=False):
        num_nodes = len(entry['processed_question_toks']) + len(entry['table_names']) + len(entry['column_names'])
        q, s = np.array(entry['relations'], dtype='<U100'), np.array(db['relations'], dtype='<U100')
        qs, sq = np.array(entry['schema_linking'][0], dtype='<U100'), np.array(entry['schema_linking'][1], dtype='<U100')
        adjacency = np.concatenate([
            np.concatenate([q, q_s], axis=1),
            np.concatenate([s_q, s], axis=1)
        ], axis=0).flatten().tolist()
        filter_relations = ['question-question', 'table-table', 'column-column',
            'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable', 'table-column', 'column-table',
            'table-question-nomatch', 'question-table-nomatch', 'column-question-nomatch', 'question-column-nomatch',
            'question-question-dist0', 'table-table-identity', 'column-column-identity']
        edges = [(idx // num_nodes, idx % num_nodes, r) for idx, r in enumerate(adjacency) if r not in filter_relations]
        entry['edges'] = edges
        return entry
    
    def rgcn(self, entry, db, verbose=False):
        pass

    def rgat(self, entry, db, verbose=False):
        pass