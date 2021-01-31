#coding=utf8
import os, pickle, json
import torch
import numpy as np
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from utils.constants import BOS, UNK, GRAMMAR_FILEPATH, SCHEMA_TYPES, RELATIONS
from utils.graph_utils import GraphFactory
from utils.vocab import Vocab
from utils.word2vec import Word2vecUtils
from transformers import AutoTokenizer
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, ptm=None, method='lgnn', table_path='data/tables.bin', tables=None, add_cls=False, fast=True, position='qtc'):
        cls.ptm = ptm
        cls.grammar = ASDLGrammar.from_filepath(GRAMMAR_FILEPATH)
        cls.trans = TransitionSystem.get_class_by_lang('sql')(cls.grammar)
        cls.tables = pickle.load(open(table_path, 'rb')) if tables is None else tables
        cls.evaluator = Evaluator(cls.trans)
        if ptm is None:
            cls.word2vec = Word2vecUtils()
            cls.tokenizer = lambda x: x
            cls.word_vocab = Vocab(padding=True, unk=True, boundary=True, default=UNK,
                filepath='./pretrained_models/glove-42b-300d/vocab.txt', specials=SCHEMA_TYPES) # word vocab for glove.42B.300d
        else:
            cls.tokenizer = AutoTokenizer.from_pretrained(os.path.join('./pretrained_models', ptm))
            cls.word_vocab = cls.tokenizer.get_vocab()
        cls.add_cls = add_cls # whether add special CLS node
        cls.relation_vocab = Vocab(padding=False, unk=False, boundary=False, iterable=RELATIONS, default=None)
        cls.fast, cls.position = fast, position
        cls.graph_factory = GraphFactory(method, cls.add_cls, cls.relation_vocab)

    @classmethod
    def load_dataset(cls, choice, debug=False):
        assert choice in ['train', 'dev']
        fp = os.path.join('data', choice + '.dgl.bin') if cls.fast else \
            os.path.join('data', choice + '.bin')
        datasets = pickle.load(open(fp, 'rb'))
        # question_lens = [len(ex['processed_question_toks']) for ex in datasets]
        # print('Max/Min/Avg question length in %s dataset is: %d/%d/%.2f' % (choice, max(question_lens), min(question_lens), float(sum(question_lens))/len(question_lens)))
        # action_lens = [len(ex['actions']) for ex in datasets]
        # print('Max/Min/Avg action length in %s dataset is: %d/%d/%.2f' % (choice, max(action_lens), min(action_lens), float(sum(action_lens))/len(action_lens)))
        examples, outliers = [], 0
        for ex in datasets:
            if choice == 'train' and len(cls.tables[ex['db_id']]['processed_column_toks']) > 100:
                outliers += 1
                continue
            examples.append(cls(ex, cls.tables[ex['db_id']]))
            if debug and len(examples) >= 100:
                return examples
        if choice == 'train':
            print("Skip %d extremely large samples in training dataset ..." % (outliers))
        return examples

    def __init__(self, ex: dict, db: dict):
        super(Example, self).__init__()
        self.ex = ex
        self.db = db

        """ Mapping word to corresponding index """
        if Example.ptm is None:
            self.question = [BOS] + ex['processed_question_toks'] if Example.add_cls else ex['processed_question_toks']
            self.question_id = [Example.word_vocab[w] for w in self.question]
            self.column = [[db['column_types'][idx].lower()] + c for idx, c in enumerate(db['processed_column_toks'])]
            self.column_id = [[Example.word_vocab[w] for w in c] for c in self.column]
            self.table = [['table'] + t for t in db['processed_table_toks']]
            self.table_id = [[Example.word_vocab[w] for w in t] for t in self.table]
        else:
            t = Example.tokenizer
            self.question = [t.cls_token] + [q.lower() for q in ex['raw_question_toks']] if Example.add_cls else \
                [q.lower() for q in ex['raw_question_toks']]
            self.question_id = [] if Example.add_cls else [t.cls_token_id] # map token to id
            self.question_mask_ptm = [] # remove SEP token in our case
            self.question_subword_len = [] # subword len for each word, exclude SEP token
            for w in self.question:
                toks = t.convert_tokens_to_ids(t.tokenize(w))
                self.question_id.extend(toks)
                self.question_subword_len.append(len(toks))
            self.question_mask_ptm = [1] * len(self.question_id) + [0] if Example.add_cls else \
                [0] + [1] * (len(self.question_id) - 1) + [0]
            self.question_id.append(t.sep_token_id)

            self.table = [['table'] + t.lower().split() for t in db['table_names']]
            self.table_id, self.table_mask_ptm, self.table_subword_len = [], [], []
            table_word_len = []
            for s in self.table:
                l = 0
                for w in s:
                    toks = t.convert_tokens_to_ids(t.tokenize(w))
                    self.table_id.extend(toks)
                    self.table_subword_len.append(len(toks))
                    l += len(toks)
                table_word_len.append(l)
            self.table_mask_ptm = [1] * len(self.table_id) + [0]
            self.table_id.append(t.sep_token_id)

            self.column = [[db['column_types'][idx].lower()] + c.lower().split() for idx, (_, c) in enumerate(db['column_names'])]
            self.column_id, self.column_mask_ptm, self.column_subword_len = [], [], []
            column_word_len = []
            for s in self.column:
                l = 0
                for w in s:
                    toks = t.convert_tokens_to_ids(t.tokenize(w))
                    self.column_id.extend(toks)
                    self.column_subword_len.append(len(toks))
                    l += len(toks)
                column_word_len.append(l)
            self.column_mask_ptm = [1] * len(self.column_id) + [0]
            self.column_id.append(t.sep_token_id)

            self.input_id = self.question_id + self.table_id + self.column_id
            self.segment_id = [0] * len(self.question_id) + [1] * (len(self.table_id) + len(self.column_id)) \
                if Example.ptm != 'grappa_large_jnt' and not Example.ptm.startswith('roberta') \
                else [0] * (len(self.question_id) + len(self.table_id) + len(self.column_id))

            # by default, format: [CLS] q1 q2 ... [SEP] t1 t2 ... [SEP] c1 c2 ... [SEP]
            if Example.position == 'qtc':
                self.position_id = list(range(len(self.input_id)))
            else:
                # another choice: [CLS] q1 q2 ... [SEP] * [SEP] t1 c1 c2 ... t2 c3 c4 ... [SEP]
                question_position_id = list(range(len(self.question_id)))
                start = len(question_position_id)
                column_position_id = [start + i for i in range(column_word_len[0])] # special symbol *
                start += column_word_len[0]
                column_position_id.insert(0, start) # add intermediate [SEP]
                start += 1
                table_position_id = []
                for idx, col_idxs in enumerate(db['table2columns']):
                    table_position_id.extend([start + i for i in range(table_word_len[idx])])
                    start += table_word_len[idx]
                    for col_id in col_idxs:
                        column_position_id.extend([start + i for i in range(column_word_len[col_id])])
                        start += column_word_len[col_id]
                column_position_id.append(start) # last [SEP]
                self.position_id = question_position_id + table_position_id + column_position_id
                assert len(self.position_id) == len(self.input_id)

            self.question_mask_ptm = self.question_mask_ptm + [0] * (len(self.table_id) + len(self.column_id))
            self.table_mask_ptm = [0] * len(self.question_id) + self.table_mask_ptm + [0] * len(self.column_id)
            self.column_mask_ptm = [0] * (len(self.question_id) + len(self.table_id)) + self.column_mask_ptm

        self.graph = Example.graph_factory.graph_construction(ex, db, fast=Example.fast)

        # outputs
        self.ast = ex['ast']
        self.query = ' '.join(ex['query'].split('\t'))
        self.tgt_action = ex['actions']
        self.used_tables, self.used_columns = ex['used_tables'], ex['used_columns']
