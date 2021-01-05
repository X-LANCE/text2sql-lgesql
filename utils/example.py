#coding=utf8
import os, pickle, json
import torch
import numpy as np
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from preprocess.dataset_process import process_tables, process_example
from utils.vocab import BOS, Vocab, RELATIVE_POSITION, POS, TYPE_CONTROL
from utils.word2vec import Word2vecUtils
from transformers import AutoTokenizer

GRAMMAR_PATH = 'asdl/sql/grammar/sql_asdl_v2.txt'
TRANS_NAME = 'sql'

def str2idx(s):
    return Example.relative_position_vocab[s]

vectorize_str2idx = np.vectorize(str2idx)

class Example():

    @classmethod
    def configuration(cls, ptm=None, processed=True, tables=None, table_path='data/tables.nltk.bin',
            vocab_path='data/vocab.nltk.txt', add_cls=False, test=False):
        cls.grammar = ASDLGrammar.from_filepath(GRAMMAR_PATH)
        cls.trans = TransitionSystem.get_class_by_lang(TRANS_NAME)(cls.grammar)
        cls.processed, cls.test = processed, test
        cls.tables = tables if tables is not None else pickle.load(open(table_path, 'rb')) \
            if cls.processed else process_tables(json.load(open(table_path, 'r')))
        if not test: # use our defined evaluator
            try:
                from evaluator.evaluator import Evaluator
                cls.evaluator = Evaluator(cls.trans, cls.tables) if not test else None
            except Exception:
                print('Error while loading evaluator')
                exit(0)
        cls.ptm = ptm
        cls.add_cls = add_cls #if ptm is None else True
        if ptm is None:
            cls.word_vocab = Vocab(padding=True, unk=True, bos=True, file_path=vocab_path, specials=[
                'TYPE[table]', 'TYPE[others]', 'TYPE[text]', 'TYPE[time]', 'TYPE[number]', 'TYPE[boolean]'
            ]) # word vocab for question and schema items
            # cls.type_vocab = Vocab(padding=False, unk=False, bos=False, items=TYPE_CONTROL, default=None, specials=[
                # 'TYPE[table]', 'TYPE[others]', 'TYPE[text]', 'TYPE[time]', 'TYPE[number]', 'TYPE[boolean]'
            # ]) # UNK is used for punctuations
            cls.word2vec = Word2vecUtils()
        else:
            cls.tokenizer = AutoTokenizer.from_pretrained(os.path.join('./pretrained_models', ptm))
            cls.word_vocab = []
        if cls.add_cls:
            CLS_RELATION = ['cls-identity', 'cls-question', 'question-cls', 'cls-table', 'table-cls', 'cls-column', 'column-cls']
            cls.relative_position_vocab = Vocab(padding=False, unk=False, bos=False, items=RELATIVE_POSITION + CLS_RELATION, default=None, specials=[])
        else:
            cls.relative_position_vocab = Vocab(padding=False, unk=False, bos=False, items=RELATIVE_POSITION, default=None, specials=[])

    @classmethod
    def load_dataset(cls, choice, debug=False):
        assert choice in ['train', 'dev']
        if cls.processed:
            fp = os.path.join('data', choice + '.nltk.bin')
            datasets = pickle.load(open(fp, 'rb'))
        else:
            fp = os.path.join('data', choice + '.json')
            old_datasets = json.load(open(fp, 'r'))
            datasets = []
            for entry in old_datasets:
                # only lazy preprocessing for evaluation dataset (dev and test), no cross domain schema linking
                # training dataset need to be processed in advance
                datasets.append(process_example(entry, cls.tables, cls.trans, cross_database=False))
        question_lens = [len(ex['processed_question_toks']) for ex in datasets]
        print('Max/Min/Avg question length in %s dataset is: %d/%d/%.2f' % (choice, max(question_lens), min(question_lens), float(sum(question_lens))/len(question_lens)))
        action_lens = [len(ex['actions']) for ex in datasets]
        print('Max/Min/Avg action length in %s dataset is: %d/%d/%.2f' % (choice, max(action_lens), min(action_lens), float(sum(action_lens))/len(action_lens)))
        examples, outliers = [], 0
        for ex in datasets:
            if choice == 'train' and len(cls.tables[ex['db_id']]['processed_column_toks']) > 100:
                outliers += 1
                continue
            examples.append(cls(ex, cls.tables[ex['db_id']]))
            if choice == 'train' and debug and len(examples) >= 100:
                return examples
        if choice == 'train':
            print("Skip %d extremely large samples in training dataset ..." % (outliers))
        return examples

    def __init__(self, ex: dict, db: dict):
        super(Example, self).__init__()
        self.ex = ex
        self.db = db
        self.relation_dropout = False

        """ Mapping word to corresponding index """
        if Example.ptm is None:
            self.question = [BOS] + ex['processed_question_toks'] if Example.add_cls else ex['processed_question_toks']
            self.question_id = [Example.word_vocab[w] for w in self.question]
            # self.question_type = ex['type_control']
            # self.question_type_id = [Example.type_vocab[t] for t in self.question_type]

            # short cut for schema items in corresponding database
            self.column = [['TYPE[' + db['column_types'][idx].lower() + ']'] + c for idx, c in enumerate(db['processed_column_toks'])]
            # self.column = db['processed_column_toks']
            self.column_id = [[Example.word_vocab[w] for w in c] for c in self.column]
            # self.column_type = ['TYPE[' + t.lower() + ']' for t in db['column_types']]
            # self.column_type_id = [Example.type_vocab[t] for t in self.column_type]
            self.table = [['TYPE[table]'] + t for t in db['processed_table_toks']]
            # self.table = db['processed_table_toks']
            self.table_id = [[Example.word_vocab[w] for w in t] for t in self.table]
            # self.table_type = ['TYPE[table]' for _ in range(len(self.table))]
            # self.table_type_id = [Example.type_vocab[t] for t in self.table_type]
        else:
            t = Example.tokenizer
            self.question = [t.cls_token] + [q.lower() for q in ex['raw_question_toks']] if Example.add_cls else \
                [q.lower() for q in ex['raw_question_toks']]
            # self.question = [t.cls_token] + ex['processed_question_toks']
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
            # self.question_position_id = list(range(len(self.question_id)))

            # self.table = [t.lower().split() for t in db['table_names']]
            self.table = [['table'] + t.lower().split() for t in db['table_names']]
            # self.table = [['table'] + t for t in db['processed_table_toks']]
            # self.table = db['processed_table_toks']
            self.table_id, self.table_position_id = [], []
            self.table_mask_ptm, self.table_subword_len, self.table_word_len = [], [], []
            for s in self.table:
                l = 0
                for w in s:
                    toks = t.convert_tokens_to_ids(t.tokenize(w))
                    self.table_id.extend(toks)
                    self.table_subword_len.append(len(toks))
                    l += len(toks)
                self.table_word_len.append(l)
                # self.table_id.append(t.sep_token_id) # add SEP token after each table
                # self.table_position_id.extend(list(range(l + 1)))
                # self.table_mask_ptm.extend([1] * l + [0])
            self.table_mask_ptm = [1] * len(self.table_id) + [0]
            self.table_id.append(t.sep_token_id)

            # self.column = [c.lower().split() for _, c in db['column_names']]
            self.column = [[db['column_types'][idx].lower()] + c.lower().split() for idx, (_, c) in enumerate(db['column_names'])]
            # self.column = [[db['column_types'][idx].lower()] + c for idx, c in enumerate(db['processed_column_toks'])]
            # self.column = db['processed_column_toks']
            self.column_id, self.column_position_id = [], []
            self.column_mask_ptm, self.column_subword_len, self.column_word_len = [], [], []
            for s in self.column:
                l = 0
                for w in s:
                    toks = t.convert_tokens_to_ids(t.tokenize(w))
                    self.column_id.extend(toks)
                    self.column_subword_len.append(len(toks))
                    l += len(toks)
                self.column_word_len.append(l)
                # self.column_id.append(t.sep_token_id)
                # self.column_position_id.extend(list(range(l + 1)))
                # self.column_mask_ptm.extend([1] * l + [0])
            self.column_mask_ptm = [1] * len(self.column_id) + [0]
            self.column_id.append(t.sep_token_id)

            self.input_id = self.question_id + self.table_id + self.column_id
            self.segment_id = [0] * len(self.question_id) + [1] * (len(self.table_id) + len(self.column_id)) if not Example.ptm.startswith('roberta')\
                else [0] * (len(self.question_id) + len(self.table_id) + len(self.column_id))

            # re scatter position id <cls> q1 q2 <sep> * t1 c1 c2 t2 c3 c4 <sep>
            # question_position_id, table_position_id = list(range(len(self.question_id))), []
            # start = len(question_position_id)
            # column_position_id = [start + i for i in range(self.column_word_len[0])]
            # start += self.column_word_len[0]
            # for idx, col_idxs in enumerate(db['table2columns']):
                # table_position_id.extend([start + i for i in range(self.table_word_len[idx])])
                # start += self.table_word_len[idx]
                # for col_id in col_idxs:
                    # column_position_id.extend([start + i for i in range(self.column_word_len[col_id])])
                    # start += self.column_word_len[col_id]
            # column_position_id.append(start)
            # self.position_id = question_position_id + table_position_id + column_position_id
            # assert len(self.position_id) == len(self.input_id)

            # self.position_id = self.question_position_id + self.table_position_id + self.column_position_id
            self.mask_ptm = self.question_mask_ptm + self.table_mask_ptm + self.column_mask_ptm
            self.question_mask_ptm = self.question_mask_ptm + [0] * (len(self.table_id) + len(self.column_id))
            self.table_mask_ptm = [0] * len(self.question_id) + self.table_mask_ptm + [0] * len(self.column_id)
            self.column_mask_ptm = [0] * (len(self.question_id) + len(self.table_id)) + self.column_mask_ptm

        q = np.array(ex['relations'], dtype='<U100')
        s = np.array(db['relations'], dtype='<U100')
        q_s, s_q = ex['schema_linking'][db['db_id']]
        q_s, s_q = np.array(q_s, dtype='<U100'), np.array(s_q, dtype='<U100')
        if Example.add_cls:
            # add CLS for schema linking
            cls_cls = np.array(['cls-identity'], dtype='<U100')[np.newaxis, :]
            cls_q = np.array(['cls-question'] * q.shape[0], dtype='<U100')[np.newaxis, :]
            q_cls = np.array(['question-cls'] * q.shape[0], dtype='<U100')[:, np.newaxis]
            cls_s = np.array(['cls-table'] * len(self.table) + ['cls-column'] * len(self.column), dtype='<U100')[np.newaxis, :]
            s_cls = np.array(['table-cls'] * len(self.table) + ['column-cls'] * len(self.column), dtype='<U100')[:, np.newaxis]
            self.relation = np.concatenate([
                np.concatenate([cls_cls, cls_q, cls_s], axis=1),
                np.concatenate([q_cls, q, q_s], axis=1),
                np.concatenate([s_cls, s_q, s], axis=1)
            ], axis=0)
        else:
            self.relation = np.concatenate([
                np.concatenate([q, q_s], axis=1),
                np.concatenate([s_q, s], axis=1)
            ], axis=0)
        self.relation_id = vectorize_str2idx(self.relation)


        self.table_min_distance, self.column_min_distance = np.array(ex['table_min_distance']), np.array(ex['column_min_distance'])
        self.table_sem_similarity, self.column_sem_similarity = np.array(ex['table_sem_similarity']), np.array(ex['column_sem_similarity'])
        table_score = self.table_sem_similarity - self.table_min_distance
        column_score = self.column_sem_similarity - self.column_min_distance
        self.table_sampling_prob, self.column_sampling_prob = np.power(2., table_score).tolist(), np.power(2., column_score).tolist()
        table_ls = np.where(self.table_min_distance < 0.5, 1.0, (5 - self.table_min_distance) * 0.05)
        column_ls = np.where(self.column_min_distance < 0.5, 1.0, (5 - self.column_min_distance) * 0.05)
        self.ls = np.concatenate([table_ls, column_ls], axis=0)

        # outputs
        self.ast = ex['ast']
        self.tgt_action = ex['actions']
        self.used_tables, self.used_columns = ex['used_tables'], ex['used_columns']
