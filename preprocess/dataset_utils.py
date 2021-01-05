#coding=utf8
"""
    # @Time    : 2020/3/22
    # @Author  : Ruisheng Cao
    # @File    : dataset_utils.py
    # @Software: VScode
"""
import os, json, re, copy, sqlite3
import numpy as np
import stanza, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from utils.vocab import MAX_RELATIVE_DIST
from itertools import product, combinations
import networkx as nx
from networkx.algorithms.shortest_paths.generic import has_path, shortest_path_length
from transformers.tokenization_utils import _is_whitespace, _is_control, _is_punctuation

NLP = stanza.Pipeline('en', processors='tokenize,pos,lemma') #, tokenize_pretokenized=True)

STOPWORDS = stopwords.words("english")
def skip_special_symbols(s):
    if s.lower() in STOPWORDS:
        return True
    elif len(s) == 1:
        return bool(_is_whitespace(s) | _is_control(s) | _is_punctuation(s))
    else:
        return False

def map_relation_to_score(a):
    if 'nomatch' in a:
        return 0.
    elif 'exactmatch' in a:
        return 1.
    elif 'partialmatch' in a:
        return .5
    elif 'value' in a:
        return 1.
    else:
        raise ValueError('Something error while parsing schema linking infomation into scores.')
relation2score = np.vectorize(map_relation_to_score)

def question_normalization(q):
    quotation_marks = ['""', '"', "''", "``", "`", "’’", "‘‘", "‘", "’", "“", "”"]
    for m in quotation_marks:
        q = q.replace(m, "'")
    replacements = { "'.": "' .", "'?": "' ?" }
    for k, v in replacements.items():
        q = q.replace(k, v)
    old, new = r"'([a-zA-Z])'", r"' \1 '"
    q = re.sub(old, new, q)
    return q

AGG = ['average', 'sum', 'max', 'min', 'minimum', 'maximum', 'between']
def question_type_control(question_toks, pos_tags):
    type_control = []
    for w, t in zip(question_toks, pos_tags):
        if w.lower() in AGG:
            type_control.append('AGG')
        elif t.upper() in ['RBR', 'JJR']:
            type_control.append('MORE')
        elif t.upper() in ['RBS', 'JJS']:
            type_control.append('MOST')
        elif is_year(w):
            type_control.append('YEAR')
        elif is_number(w):
            type_control.append('NUMBER')
        else:
            type_control.append('NONE')
    return type_control

def quote_normalization(question):
    """ Normalize all usage of quotation marks into a separate ' """
    new_question, quotation_marks = [], ["'", '"', '`', '‘', '’', '“', '”', '``', "''", "‘‘", "’’"]
    for idx, tok in enumerate(question):
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += ["'", tok[1:-1], "'"]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += ["'", tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], "'" ]
        elif tok in quotation_marks:
            new_question.append("'")
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += ["'", tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
    return new_question

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class Preprocessor():

    def __init__(self, db_dir='data/database'):
        super(Preprocessor, self).__init__()
        self.lemma = WordNetLemmatizer()
        self.db_dir = db_dir

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks, table_names = [], []
        for tab in db['table_names']:
            # nltk_result = nltk.pos_tag(nltk.word_tokenize(tab))
            # tab = [self.lemma.lemmatize(x, pos=get_wordnet_pos(t)).lower() for x, t in nltk_result]
            doc = NLP(tab)
            tab = [w.lemma.lower() for s in doc.sentences for w in s.words]
            table_toks.append(tab)
            table_names.append(" ".join(tab))
        db['processed_table_toks'], db['processed_table_names'] = table_toks, table_names
        column_toks, column_names = [], []
        for _, c in db['column_names']:
            # nltk_result = nltk.pos_tag(nltk.word_tokenize(c))
            # c = [self.lemma.lemmatize(x, pos=get_wordnet_pos(t)).lower() for x, t in nltk_result]
            doc = NLP(c)
            c = [w.lemma.lower() for s in doc.sentences for w in s.words]
            column_toks.append(c)
            column_names.append(" ".join(c))
        db['processed_column_toks'], db['processed_column_names'] = column_toks, column_names
        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [[] for _ in range(len(table_names))] # from table id to column ids list
        for col_id, col in enumerate(db['column_names']):
            if col_id == 0: continue
            table2columns[col[0]].append(col_id)
        db['column2table'], db['table2columns'] = column2table, table2columns

        # column closure according to foreign key constraints
        set_list = []
        def keys_in_set_list(k1, k2):
            for s in set_list:
                if k1 in s or k2 in s:
                    return s
            new_s = set()
            set_list.append(new_s)
            return new_s
        foreign_keys = db['foreign_keys']
        for k1, k2 in foreign_keys:
            s = keys_in_set_list(k1, k2)
            s.add(k1)
            s.add(k2)
        db['column_closure'] = [sorted(list(s)) for s in set_list]

        t_num, c_num, dtype = len(db['table_names']), len(db['column_names']), '<U100'
        # relations in tables, tab_num * tab_num
        tab_mat = np.array([['table-table'] * t_num for _ in range(t_num)], dtype=dtype)
        table_fks = set(map(lambda pair: (column2table[pair[0]], column2table[pair[1]]), db['foreign_keys']))
        for (tab1, tab2) in table_fks:
            if (tab2, tab1) in table_fks:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'foreign-key-tab-b', 'foreign-key-tab-b'
            else:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'foreign-key-tab-f', 'foreign-key-tab-r'
        tab_mat[list(range(t_num)), list(range(t_num))] = 'table-identity'
        # relations in columns, c_num * c_num
        col_mat = np.array([['column-column'] * c_num for _ in range(c_num)], dtype=dtype)
        for i in range(t_num):
            col_ids = [idx for idx, t in enumerate(column2table) if t == i]
            col1, col2 = list(zip(*list(product(col_ids, col_ids))))
            col_mat[col1, col2] = 'same-table'
        # col_mat[0, list(range(c_num))] = '*-column'
        # col_mat[list(range(c_num)), 0] = 'column-*'
        col_mat[list(range(c_num)), list(range(c_num))] = 'column-identity'
        if len(db['foreign_keys']) > 0:
            col1, col2 = list(zip(*db['foreign_keys']))
            col_mat[col1, col2], col_mat[col2, col1] = 'foreign-key-f', 'foreign-key-r'
        # relations between tables and columns, t_num*c_num and c_num*t_num
        tab_col_mat = np.array([['table-column'] * c_num for _ in range(t_num)], dtype=dtype)
        col_tab_mat = np.array([['column-table'] * t_num for _ in range(c_num)], dtype=dtype)
        cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), range(1, c_num))))) # ignore *
        col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'belongs-to-f', 'belongs-to-r'
        # col_tab_mat[0, list(range(t_num))] = '*-table'
        # tab_col_mat[list(range(t_num)), 0] = 'table-*'
        if len(db['primary_keys']) > 0:
            cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), db['primary_keys']))))
            col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'primary-key-f', 'primary-key-r'
        relations = np.concatenate([
            np.concatenate([tab_mat, tab_col_mat], axis=1),
            np.concatenate([col_tab_mat, col_mat], axis=1)
        ], axis=0)
        db['relations'] = relations.tolist()
        if verbose:
            print('Tables:', ', '.join(db['table_names']))
            print('Lemmatized:', ', '.join(table_names))
            print('Columns:', ', '.join(list(map(lambda x: x[1], db['column_names']))))
            print('Lemmatized:', ', '.join(column_names), '\n')
        return db

    def preprocess_question(self, entry: dict, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question, tables, columns """
        # stanza tokenize, lemmatize and POS tag
        # question = question_normalization(entry['question'])
        question = ' '.join(quote_normalization(entry['question_toks']))
        doc = NLP(question)
        raw_toks = [w.text for s in doc.sentences for w in s.words]
        toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
        pos_tags = [w.xpos for s in doc.sentences for w in s.words]

        # nltk tokenize, lemmatize and POS tag
        # raw_toks = quote_normalization(entry['question_toks'])
        # nltk_result = nltk.pos_tag(raw_toks)
        # toks = [self.lemma.lemmatize(w, pos=get_wordnet_pos(t)).lower() for w, t in nltk_result]
        # pos_tags = list(map(lambda x: x[1], nltk_result))

        entry['raw_question_toks'] = raw_toks
        entry['processed_question_toks'] = toks
        entry['pos_tags'] = pos_tags

        # type control: AGG, MORE, MOST, NUMBER
        entry['type_control'] = question_type_control(toks, pos_tags)

        # relations in questions, q_num * q_num
        q_num, dtype = len(toks), '<U100'
        if q_num <= MAX_RELATIVE_DIST + 1:
            dist_vec = ['question-dist-' + str(i) for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)]
            starting = MAX_RELATIVE_DIST
        else:
            dist_vec = ['question-question'] * (q_num - MAX_RELATIVE_DIST - 1) + \
                ['question-dist-' + str(i) for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
                    ['question-question'] * (q_num - MAX_RELATIVE_DIST - 1)
            starting = q_num - 1
        q_mat = np.array([dist_vec[starting - i: starting - i + q_num] for i in range(q_num)], dtype=dtype)
        entry['relations'] = q_mat.tolist()
        if verbose:
            print('Question:', entry['question'])
            print('Tokenized:', ' '.join(entry['raw_question_toks']))
            print('Lemmatized:', ' '.join(entry['processed_question_toks']))
            print('Pos tags:', ' '.join(entry['pos_tags']), '\n')
        return entry

    def calculate_sampling_prob(self, entry: dict, db: dict, verbose: bool = False):
        """ Given used_tables and used_columns key in the entry, calculate the sampling prob for each schema item in db,
        which will be used during training in the decoder of text2sql task
        """
        max_distance = 5.
        if entry['db_id'] != db['db_id']: # not the related database
            table_min_distance = np.array([max_distance] * len(db['table_names']), dtype=np.float)
            column_min_distance = np.array([max_distance] * len(db['column_names']), dtype=np.float)
        else:
            table_min_distance, column_min_distance = SchemaGraph(db).get_distance_to_golden_schema(entry['used_tables'], entry['used_columns'], max_distance=max_distance)
        schema_linking = np.array(entry['schema_linking'][db['db_id']][0], dtype='<U100')
        table_linking, column_linking = schema_linking[:, :len(db['table_names'])], schema_linking[:, len(db['table_names']):]
        table_sem_similarity = relation2score(table_linking).max(axis=0)
        column_sem_similarity = relation2score(column_linking).max(axis=0)
        entry['table_min_distance'], entry['column_min_distance'] = table_min_distance.tolist(), column_min_distance.tolist()
        entry['table_sem_similarity'], entry['column_sem_similarity'] = table_sem_similarity.tolist(), column_sem_similarity.tolist()

        if verbose:
            print('Table distance to golden schema:', entry['table_min_distance'])
            print('Table semantic similarity to question:', entry['table_sem_similarity'])
            print('Column distance to golden schema:', entry['column_min_distance'])
            print('Column semantic similarity to question:', entry['column_sem_similarity'])
        return entry

    def extract_subgraph(self, entry: dict, db: dict, verbose: bool = False):
        sql, closure = entry['sql'], db['column_closure']
        used_schema = {'table': set(), 'column': set()}
        used_schema = self.extract_subgraph_from_sql(sql, used_schema)
        entry['used_tables'] = sorted(list(used_schema['table']))
        entry['used_columns'] = sorted(list(used_schema['column']))

        table_closure, column_closure = copy.deepcopy(used_schema['table']), copy.deepcopy(used_schema['column'])
        for col_id in used_schema['column']:
            for s in closure:
                if col_id in s:
                    column_closure.update(s)
                    break
        for col_id in column_closure - used_schema['column']:
            if col_id == 0: continue
            tab_id = db['column_names'][col_id][0]
            table_closure.add(tab_id)
        table_closure, column_closure = sorted(list(table_closure)), sorted(list(column_closure))
        entry['used_tables_closure'], entry['used_columns_closure'] = table_closure, column_closure

        if verbose:
            print('Used tables:', entry['used_tables'])
            print('Used tables closure:', entry['used_tables_closure'])
            print('Used columns:', entry['used_columns'])
            print('Used columns closure:', entry['used_columns_closure'], '\n')
        return entry

    def extract_subgraph_from_sql(self, sql: dict, used_schema: dict):
        select_items = sql['select'][1]
        # select clause
        for _, val_unit in select_items:
            if val_unit[0] == 0:
                col_unit = val_unit[1]
                used_schema['column'].add(col_unit[1])
            else:
                col_unit1, col_unit2 = val_unit[1:]
                used_schema['column'].add(col_unit1[1])
                used_schema['column'].add(col_unit2[1])
        # from clause conds
        table_units = sql['from']['table_units']
        for _, t in table_units:
            if type(t) == dict:
                used_schema = self.extract_subgraph_from_sql(t, used_schema)
            else:
                used_schema['table'].add(t)
        # from, where and having conds
        used_schema = self.extract_subgraph_from_conds(sql['from']['conds'], used_schema)
        used_schema = self.extract_subgraph_from_conds(sql['where'], used_schema)
        used_schema = self.extract_subgraph_from_conds(sql['having'], used_schema)
        # groupBy and orderBy clause
        groupBy = sql['groupBy']
        for col_unit in groupBy:
            used_schema['column'].add(col_unit[1])
        orderBy = sql['orderBy']
        if len(orderBy) > 0:
            orderBy = orderBy[1]
            for val_unit in orderBy:
                if val_unit[0] == 0:
                    col_unit = val_unit[1]
                    used_schema['column'].add(col_unit[1])
                else:
                    col_unit1, col_unit2 = val_unit[1:]
                    used_schema['column'].add(col_unit1[1])
                    used_schema['column'].add(col_unit2[1])
        # union, intersect and except clause
        if sql['intersect']:
            used_schema = self.extract_subgraph_from_sql(sql['intersect'], used_schema)
        if sql['union']:
            used_schema = self.extract_subgraph_from_sql(sql['union'], used_schema)
        if sql['except']:
            used_schema = self.extract_subgraph_from_sql(sql['except'], used_schema)
        return used_schema

    def extract_subgraph_from_conds(self, conds: list, used_schema: dict):
        if len(conds) == 0:
            return used_schema
        for cond in conds:
            if cond in ['and', 'or']:
                continue
            val_unit, val1, val2 = cond[2:]
            if val_unit[0] == 0:
                col_unit = val_unit[1]
                used_schema['column'].add(col_unit[1])
            else:
                col_unit1, col_unit2 = val_unit[1:]
                used_schema['column'].add(col_unit1[1])
                used_schema['column'].add(col_unit2[1])
            if type(val1) == list:
                used_schema['column'].add(val1[1])
            elif type(val1) == dict:
                used_schema = self.extract_subgraph_from_sql(val1, used_schema)
            if type(val2) == list:
                used_schema['column'].add(val1[1])
            elif type(val2) == dict:
                used_schema = self.extract_subgraph_from_sql(val2, used_schema)
        return used_schema

    def schema_linking(self, entry: dict, db: dict, db_content: bool = True, verbose: bool = False):
        """ Perform schema linking: both question and database need to be preprocessed """
        raw_question_toks, question_toks = entry['raw_question_toks'], entry['processed_question_toks']
        table_toks, column_toks = db['processed_table_toks'], db['processed_column_toks']
        table_names, column_names = db['processed_table_names'], db['processed_column_names']
        q_num, t_num, c_num, dtype = len(question_toks), len(table_toks), len(column_toks), '<U100'
        total_num = q_num + t_num + c_num

        # relations between questions and tables, q_num*t_num and t_num*q_num
        # table_matched_pairs = {'partial': [], 'exact': []}
        q_tab_mat = np.array([['question-table-nomatch'] * t_num for _ in range(q_num)], dtype=dtype)
        tab_q_mat = np.array([['table-question-nomatch'] * q_num for _ in range(t_num)], dtype=dtype)
        max_len = max([len(t) for t in table_toks])
        index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
        for i, j in index_pairs:
            phrase = ' '.join(question_toks[i: j])
            # if skip_special_symbols(phrase): continue
            if phrase in STOPWORDS: continue
            for idx, name in enumerate(table_names):
                if phrase == name: # fully match will overwrite partial match due to sort
                    q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
                    tab_q_mat[idx, range(i, j)] = 'table-question-exactmatch'
                    # table_matched_pairs['exact'].append(str((name, idx, phrase, i, j)))
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'
                    tab_q_mat[idx, range(i, j)] = 'table-question-partialmatch'
                    # table_matched_pairs['partial'].append(str((name, idx, phrase, i, j)))

        # relations between questions and columns
        # column_matched_pairs = {'partial': [], 'exact': [], 'value': []}
        q_col_mat = np.array([['question-column-nomatch'] * c_num for _ in range(q_num)], dtype=dtype)
        col_q_mat = np.array([['column-question-nomatch'] * q_num for _ in range(c_num)], dtype=dtype)
        max_len = max([len(c) for c in column_toks])
        index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
        for i, j in index_pairs:
            phrase = ' '.join(question_toks[i: j])
            # if skip_special_symbols(phrase): continue
            if phrase in STOPWORDS: continue
            for idx, name in enumerate(column_names):
                if phrase == name: # fully match will overwrite partial match due to sort
                    q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-exactmatch'
                    # column_matched_pairs['exact'].append(str((name, idx, phrase, i, j)))
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_col_mat[range(i, j), idx] = 'question-column-partialmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-partialmatch'
                    # column_matched_pairs['partial'].append(str((name, idx, phrase, i, j)))
        if db_content:
            db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
            if not os.path.exists(db_file):
                raise ValueError('[ERROR]: database file %s not found ...' % (db_file))
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors='ignore')
            conn.execute('pragma foreign_keys=ON')
            for i, (tab_id, col_name) in enumerate(db['column_names_original']):
                if i == 0 or 'id' in column_toks[i]: # ignore * and special token 'id'
                    continue
                tab_name = db['table_names_original'][tab_id]
                try:
                    cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (col_name, tab_name))
                    cell_values = cursor.fetchall()
                    cell_values = [str(each[0]) for each in cell_values]
                    cell_values = [str(float(each)).lower().split() if is_number(each) else each.lower().split() for each in cell_values]
                    # runnning too slow: remove nltk.word_tokenize and lemmatize
                    # cell_values = [nltk.pos_tag(nltk.word_tokenize(each)) for each in cell_values]
                    # cell_values = [[self.lemma.lemmatize(w, pos=get_wordnet_pos(t)).lower() for w, t in each] for each in cell_values]
                except Exception as e:
                    print(e)
                for j, word in enumerate(raw_question_toks):
                # for j, word in enumerate(question_toks):
                    word = str(float(word)) if is_number(word) else word
                    for c in cell_values:
                        # if word.lower() in c and 'nomatch' in q_col_mat[j, i] and (not skip_special_symbols(word)):
                        if word.lower() in c and 'nomatch' in q_col_mat[j, i] and word.lower() not in STOPWORDS:
                            q_col_mat[j, i] = 'value-column'
                            col_q_mat[i, j] = 'column-value'
                            # column_matched_pairs['value'].append(str((column_names[i], i, word, j, j + 1)))
                            break
            conn.close()

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
        q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
        if 'schema_linking' not in entry:
            entry['schema_linking'] = {}
        entry['schema_linking'][db['db_id']] = (q_schema.tolist(), schema_q.tolist())

        # if verbose:
            # print('Question:', ' '.join(question_toks))
            # print('Table matched: (table name, column id, question span, start id, end id)')
            # print('Exact match:', ', '.join(table_matched_pairs['exact']) if table_matched_pairs['exact'] else 'empty')
            # print('Partial match:', ', '.join(table_matched_pairs['partial']) if table_matched_pairs['partial'] else 'empty')
            # print('Column matched: (column name, column id, question span, start id, end id)')
            # print('Exact match:', ', '.join(column_matched_pairs['exact']) if column_matched_pairs['exact'] else 'empty')
            # print('Partial match:', ', '.join(column_matched_pairs['partial']) if column_matched_pairs['partial'] else 'empty')
            # print('Value match:', ', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty', '\n')
        return entry

def get_table_connections(columns, foreign_keys):
    """
        columns is list of tuple (tab_id, col_name_string)
        foreign_keys is list of typle (column_id, refered_column_id)
        @return:
            list of tuple (tab1, tab2), if [tab1, tab2] exists, [tab2, tab1] must exist
    """
    connections = list()
    for col1, col2 in foreign_keys:
        tab1, tab2 = columns[col1][0], columns[col2][0]
        if [tab1, tab2] not in connections and tab1 != tab2:
            connections.append([tab1, tab2])
            connections.append([tab2, tab1])
    return connections

def extract_values(sql, values=set()):
    table_units = sql['from']['table_units']
    for t in table_units:
        if t[0] == 'sql':
            values = extract_values(t[1], values)
    if sql['where']:
        for cond in sql['where']:
            if cond in ['and', 'or']: continue
            val1, val2 = cond[3], cond[4]
            values = extract_value_from_val(val1, values)
            values = extract_value_from_val(val2, values)
    if sql['having']:
        for cond in sql['having']:
            if cond in ['and', 'or']: continue
            val1, val2 = cond[3], cond[4]
            values = extract_value_from_val(val1, values)
            values = extract_value_from_val(val2, values)
    if sql['limit']:
        values.add(int(sql['limit']))
    if sql['intersect']:
        values = extract_values(sql['intersect'], values)
    if sql['union']:
        values = extract_values(sql['union'], values)
    if sql['except']:
        values = extract_values(sql['except'], values)
    return values

def extract_value_from_val(val, values):
    if type(val) == list:
        pass
    elif type(val) == dict:
        return extract_values(val, values)
    elif val is not None:
        values.add(val) # str or float
    return values

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_year(s):
    if len(str(s)) == 4 and str(s).isdigit() and int(str(s)[:2]) < 22 and int(str(s)[:2]) > 15:
        return True
    return False

class SchemaGraph():

    def __init__(self, db: dict):
        super(SchemaGraph, self).__init__()
        self.db = db
        self.g = self._build_graph(db)

    def _build_graph(self, db: dict):
        g = nx.Graph()
        # add tables and columns, add bias for column idx
        g.add_nodes_from(range(len(db['table_names']) + len(db['column_names'])))
        # add table-column relations
        g.add_edges_from([(tab_id, idx + len(db['table_names'])) for idx, tab_id in enumerate(db['column2table']) if idx != 0]) # ignore column *
        g.add_edges_from([(tab_id, len(db['table_names'])) for tab_id in range(len(db['table_names']))]) # add table -> * for each table
        # add foreign key relations for table and column
        g.add_edges_from([(db['column2table'][idx[0]], db['column2table'][idx[1]]) for idx in db['foreign_keys']])
        g.add_edges_from([(idx[0] + len(db['table_names']), idx[1] + len(db['table_names'])) for idx in db['foreign_keys']])
        # add edge among columns under the same table
        for tab_id in range(len(db['table_names'])):
            columns = [col_id + len(db['table_names']) for col_id in db['table2columns'][tab_id]]
            g.add_edges_from(list(combinations(columns, 2)))
        return g

    def get_distance_to_golden_schema(self, used_tables, used_columns, max_distance=5.):
        target_schema = used_tables + [idx + len(self.db['table_names']) for idx in used_columns]
        tab_dist, col_dist = [], []
        for idx in range(len(self.db['table_names'])):
            if idx in used_tables:
                tab_dist.append(0.)
            else:
                tab_dist.append(min([shortest_path_length(self.g, idx, node) if has_path(self.g, idx, node) else max_distance for node in target_schema]))
        for idx in range(len(self.db['column_names'])):
            if idx in used_columns:
                col_dist.append(0.)
            else:
                if idx == 0:
                    col_dist.append(1.)
                    continue
                col_dist.append(min([shortest_path_length(self.g, idx + len(self.db['table_names']), node) if has_path(self.g, idx + len(self.db['table_names']), node) else max_distance for node in target_schema]))
        return np.array(tab_dist, dtype=np.float), np.array(col_dist, dtype=np.float)
