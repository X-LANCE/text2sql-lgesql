#coding=utf8
import torch
import numpy as np
from utils.example import Example
from utils.vocab import PAD, UNK, RELATIVE_POSITION
from model.model_utils import lens2mask, mask2matrix, cached_property
import torch.nn.functional as F

def get_relation_mask(q, t, c, p, mask_area='all'):
    """ q, t, c is the length of each types~(question, table, column) of nodes """
    l = q + t + c
    if mask_area == 'all':
        prob = torch.full((l, l), p, dtype=torch.float)
    else:
        # below 3 lines: only mask schema linking relations
        prob = torch.zeros((l, l), dtype=torch.float)
        prob[:q, q:] = p
        prob[q:, :q] = p
    return torch.bernoulli(prob).bool()

def obtain_generic_relation(max_q=50, max_t=50, max_c=200):
    rel2id = Example.relative_position_vocab
    qq = torch.full((max_q, max_q), rel2id['question-question'], dtype=torch.long)
    qt = torch.full((max_q, max_t), rel2id['question-table-nomatch'], dtype=torch.long)
    qc = torch.full((max_q, max_c), rel2id['question-column-nomatch'], dtype=torch.long)
    tq = torch.full((max_t, max_q), rel2id['table-question-nomatch'], dtype=torch.long)
    tt = torch.full((max_t, max_t), rel2id['table-table'], dtype=torch.long)
    tc = torch.full((max_t, max_c), rel2id['table-column'], dtype=torch.long)
    cq = torch.full((max_c, max_q), rel2id['column-question-nomatch'], dtype=torch.long)
    ct = torch.full((max_c, max_t), rel2id['column-table'], dtype=torch.long)
    cc = torch.full((max_c, max_c), rel2id['column-column'], dtype=torch.long)
    if Example.add_cls:
        # revise CLS token
        qq[0, 0] = rel2id['cls-identity']
        qt[0], tq[:, 0] = rel2id['cls-table'], rel2id['table-cls']
        qc[0], cq[:, 0] = rel2id['cls-column'], rel2id['column-cls']
    mask = torch.cat([
        torch.cat([qq, qt, qc], dim=1),
        torch.cat([tq, tt, tc], dim=1),
        torch.cat([cq, ct, cc], dim=1)
    ], dim=0)
    return mask

def from_example_list_base(ex_list, device='cpu', train=True):
    """
        Some common fields in Batch() obj:
            question_lens: torch.long, bsize
            questions: torch.long, bsize x max_question_len
            table_lens: torch.long, bsize, number of tables for each example
            table_word_lens: torch.long, number of words for each table name
            tables: torch.long, sum_of_tables x max_table_word_len
            column_lens: torch.long, bsize, number of columns for each example
            column_word_lens: torch.long, number of words for each column name
            columns: torch.long, sum_of_columns x max_column_word_len
    """
    batch = Batch(ex_list, device)
    ptm = Example.ptm
    pad_idx = Example.word_vocab[PAD] if ptm is None else Example.tokenizer.pad_token_id

    question_lens = [len(ex.question) for ex in ex_list]
    batch.question_lens = torch.tensor(question_lens, dtype=torch.long, device=device)
    if ptm is None:
        questions = [ex.question_id + [pad_idx] * (batch.max_question_len - len(ex.question_id)) for ex in ex_list]
        batch.questions = torch.tensor(questions, dtype=torch.long, device=device)
    else: # subword length, used for aggregation
        question_subword_lens = [l for ex in ex_list for l in ex.question_subword_len]
        batch.question_subword_lens = torch.tensor(question_subword_lens, dtype=torch.long, device=device)

    batch.table_lens = torch.tensor([len(ex.table) for ex in ex_list], dtype=torch.long, device=device)
    batch.table_mappings = [list(range(l)) for l in batch.table_lens.tolist()]
    batch.table_reverse_mappings = batch.table_mappings
    table_word_lens = [len(t) for ex in ex_list for t in ex.table]
    batch.table_word_lens = torch.tensor(table_word_lens, dtype=torch.long, device=device)
    if ptm is None:
        tables = [t + [pad_idx] * (batch.max_table_word_len - len(t)) for ex in ex_list for t in ex.table_id]
        batch.tables = torch.tensor(tables, dtype=torch.long, device=device)
    else: # subword lens, used for aggregation
        table_subword_lens = [l for ex in ex_list for l in ex.table_subword_len]
        batch.table_subword_lens = torch.tensor(table_subword_lens, dtype=torch.long, device=device)

    batch.column_lens = torch.tensor([len(ex.column) for ex in ex_list], dtype=torch.long, device=device)
    batch.column_mappings = [list(range(l)) for l in batch.column_lens.tolist()]
    batch.column_reverse_mappings = batch.column_mappings
    column_word_lens = [len(c) for ex in ex_list for c in ex.column]
    batch.column_word_lens = torch.tensor(column_word_lens, dtype=torch.long, device=device)
    if ptm is None:
        columns = [c + [pad_idx] * (batch.max_column_word_len - len(c)) for ex in ex_list for c in ex.column_id]
        batch.columns = torch.tensor(columns, dtype=torch.long, device=device)
    else: # subword lens, used for aggregation
        column_subword_lens = [l for ex in ex_list for l in ex.column_subword_len]
        batch.column_subword_lens = torch.tensor(column_subword_lens, dtype=torch.long, device=device)

    if ptm is not None:
        # prepare inputs for pretrained models
        batch.inputs = {"input_ids": None, "attention_mask": None, "token_type_ids": None, "position_ids": None}
        input_ids = [ex.input_id for ex in ex_list]
        input_lens = [len(ex) for ex in input_ids]
        max_len = max(input_lens)
        input_ids = [ex + [pad_idx] * (max_len - len(ex)) for ex in input_ids]
        batch.inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = [[1] * l + [0] * (max_len - l) for l in input_lens]
        batch.inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.float, device=device)
        token_type_ids = [ex.segment_id + [0] * (max_len - len(ex.segment_id)) for ex in ex_list]
        batch.inputs["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long, device=device)
        # position_ids = [ex.position_id + [0] * (max_len - len(ex.position_id)) for ex in ex_list]
        # batch.inputs["position_ids"] = torch.tensor(position_ids, dtype=torch.long, device=device)
        # extract representations after pretrained language model
        question_mask_ptm = [ex.question_mask_ptm + [0] * (max_len - len(ex.question_mask_ptm)) for ex in ex_list]
        batch.question_mask_ptm = torch.tensor(question_mask_ptm, dtype=torch.bool, device=device)
        table_mask_ptm = [ex.table_mask_ptm + [0] * (max_len - len(ex.table_mask_ptm)) for ex in ex_list]
        batch.table_mask_ptm = torch.tensor(table_mask_ptm, dtype=torch.bool, device=device)
        column_mask_ptm = [ex.column_mask_ptm + [0] * (max_len - len(ex.column_mask_ptm)) for ex in ex_list]
        batch.column_mask_ptm = torch.tensor(column_mask_ptm, dtype=torch.bool, device=device)

    if not train and ptm is None:
        # during evaluation, for words not in vocab but in glove vocab, extract its correpsonding embedding
        word2vec, unk_idx = Example.word2vec, Example.word_vocab[UNK]
        batch.question_unk_mask, batch.table_unk_mask, batch.column_unk_mask = None, None, None

        question_unk_mask = (batch.questions == unk_idx).cpu()
        if question_unk_mask.any().item():
            raw_questions = np.array([ex.question + [PAD] * (batch.max_question_len - len(ex.question)) for ex in ex_list], dtype='<U100')
            unk_words = raw_questions[question_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.question_unk_mask = question_unk_mask.masked_scatter_(torch.clone(question_unk_mask), oov_flag).to(device)
                batch.question_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)

        table_unk_mask = (batch.tables == unk_idx).cpu()
        if table_unk_mask.any().item():
            raw_tables = np.array([t + [PAD] * (batch.max_table_word_len - len(t)) for ex in ex_list for t in ex.table], dtype='<U100')
            unk_words = raw_tables[table_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.table_unk_mask = table_unk_mask.masked_scatter_(torch.clone(table_unk_mask), oov_flag).to(device)
                batch.table_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)

        column_unk_mask = (batch.columns == unk_idx).cpu()
        if column_unk_mask.any().item():
            raw_columns = np.array([c + [PAD] * (batch.max_column_word_len - len(c)) for ex in ex_list for c in ex.column], dtype='<U100')
            unk_words = raw_columns[column_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.column_unk_mask = column_unk_mask.masked_scatter_(torch.clone(column_unk_mask), oov_flag).to(device)
                batch.column_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)
    else:
        batch.question_unk_mask, batch.table_unk_mask, batch.column_unk_mask = None, None, None
    return batch

def from_example_list_ratsql(ex_list, device='cpu', train=True, **kwargs):
    """ New fields: batch.lens, batch.max_len, batch.relations, batch.relations_mask
    """
    batch = from_example_list_base(ex_list, device, train)
    pad_idx = Example.relative_position_vocab['no-relation']
    mask_prob = kwargs.pop('mask_prob', 0.)
    if mask_prob > 0.:
        relations = []
        for i, ex in enumerate(ex_list):
            rel = torch.tensor(ex.relation_id, dtype=torch.long)
            if ex.relation_dropout: # a copy of the original dataset for relation dropout
                q, t, c = batch.question_lens[i].item(), batch.table_lens[i].item(), batch.column_lens[i].item()
                mask = get_relation_mask(q, t, c, mask_prob, kwargs['mask_area'])
                if mask.any().item():
                    assert q <= Batch.max_q and t <= Batch.max_t and c <= Batch.max_c
                    index = list(range(q)) + list(range(Batch.max_q, Batch.max_q + t)) + list(range(Batch.max_q + Batch.max_t, Batch.max_q + Batch.max_t + c))
                    fill_rel = Batch.generic_relation[index, :][:, index].masked_select(mask)
                    rel = rel.masked_scatter_(mask, fill_rel)
            relations.append(F.pad(rel, (0, batch.max_len - rel.shape[0], 0, batch.max_len - rel.shape[0]), value=pad_idx))
    else:
        relations = [ F.pad(torch.tensor(ex.relation_id, dtype=torch.long),
            (0, batch.max_len - ex.relation_id.shape[0], 0, batch.max_len - ex.relation_id.shape[0]),
            value=pad_idx) for ex in ex_list ]
    batch.relations = torch.stack(relations, dim=0).to(device)
    batch.relations_mask = batch.relations == pad_idx
    batch.relations_mask_hard = (batch.relations == pad_idx) \
            | (batch.relations == Example.relative_position_vocab['column-column']) \
            | (batch.relations == Example.relative_position_vocab['table-table']) \
            | (batch.relations == Example.relative_position_vocab['table-column']) \
            | (batch.relations == Example.relative_position_vocab['column-table'])
    if train:
        batch.max_action_num = max([len(ex.tgt_action) for ex in ex_list])
    return batch

def from_example_list_ratsql_golden_decode(ex_list, device='cpu', train=True):
    """ Encoding network uses full schema, but after encoding, we only select golden schema items to decode.
    Including decoder state initialization, attention calculation and select table/column actions. Require additional fields in Batch obj:
        batch.golden_table_lens, batch.golden_column_lens, batch.golden_index
    """
    batch = from_example_list_ratsql(ex_list, device=device, train=train)
    select_tables = [ex.used_tables for ex in ex_list]
    golden_table_lens = [len(ex) for ex in select_tables]
    batch.golden_table_lens = torch.tensor(golden_table_lens, dtype=torch.long, device=device)
    batch.table_reverse_mappings = select_tables # map local id to global id
    batch.table_mappings = [dict() for _ in range(len(batch))] # map global id to local id
    for e_id in range(len(batch)):
        for idx, tab_id in enumerate(select_tables[e_id]):
            batch.table_mappings[e_id][tab_id] = idx

    select_columns = [ex.used_columns for ex in ex_list]
    golden_column_lens = [len(ex) for ex in select_columns]
    batch.golden_column_lens = torch.tensor(golden_column_lens, dtype=torch.long, device=device)
    batch.column_reverse_mappings = select_columns
    batch.column_mappings = [dict() for _ in range(len(batch))]
    for e_id in range(len(batch)):
        for idx, col_id in enumerate(select_columns[e_id]):
            batch.column_mappings[e_id][col_id] = idx

    golden_index, seqlen = [], batch.mask_split.size(1)
    for e_id in range(len(batch)):
        question_index = [e_id * seqlen + i for i in range(batch.question_lens[e_id].item())]
        table_index = [e_id * seqlen + batch.max_question_len + i for i in select_tables[e_id]]
        column_index = [e_id * seqlen + batch.max_question_len + batch.max_table_len + i for i in select_columns[e_id]]
        golden_index.extend(question_index + table_index + column_index)
    batch.golden_index = torch.tensor(golden_index, dtype=torch.long, device=device)
    return batch

def from_example_list_ratsql_golden_schema(ex_list, device='cpu', train=True):
    """ Provide golden schema as input graph for both encoding and decoding """
    batch = Batch(ex_list, device)
    pad_idx = Example.word_vocab[PAD]

    question_lens = [len(ex.question) for ex in ex_list]
    batch.question_lens = torch.tensor(question_lens, dtype=torch.long, device=device)
    questions = [ex.question_id + [pad_idx] * (batch.max_question_len - len(ex.question_id)) for ex in ex_list]
    batch.questions = torch.tensor(questions, dtype=torch.long, device=device)

    select_tables = [ex.used_tables for ex in ex_list]
    batch.table_lens = torch.tensor([len(ex) for ex in select_tables], dtype=torch.long, device=device)
    table_word_lens = [len(t) for e_id, ex in enumerate(ex_list) for t_id, t in enumerate(ex.table) if t_id in select_tables[e_id]]
    batch.table_word_lens = torch.tensor(table_word_lens, dtype=torch.long, device=device)
    tables = [t + [pad_idx] * (batch.max_table_word_len - len(t)) for e_id, ex in enumerate(ex_list) for t_id, t in enumerate(ex.table_id) if t_id in select_tables[e_id]]
    batch.tables = torch.tensor(tables, dtype=torch.long, device=device)
    batch.table_mappings = [dict() for _ in range(len(batch))] # map global id to local id
    for e_id in range(len(batch)):
        for idx, tab_id in enumerate(select_tables[e_id]):
            batch.table_mappings[e_id][tab_id] = idx
    batch.table_reverse_mappings = select_tables # map local id to global id

    select_columns = [ex.used_columns for ex in ex_list]
    batch.column_lens = torch.tensor([len(ex) for ex in select_columns], dtype=torch.long, device=device)
    column_word_lens = [len(c) for e_id, ex in enumerate(ex_list) for c_id, c in enumerate(ex.column) if c_id in select_columns[e_id]]
    batch.column_word_lens = torch.tensor(column_word_lens, dtype=torch.long, device=device)
    columns = [c + [pad_idx] * (batch.max_column_word_len - len(c)) for e_id, ex in enumerate(ex_list) for c_id, c in enumerate(ex.column_id) if c_id in select_columns[e_id]]
    batch.columns = torch.tensor(columns, dtype=torch.long, device=device)
    batch.column_mappings = [dict() for _ in range(len(batch))] # map global id to local id
    for e_id in range(len(batch)):
        for idx, col_id in enumerate(select_columns[e_id]):
            batch.column_mappings[e_id][col_id] = idx
    batch.column_reverse_mappings = select_columns # map local id to global id

    if train:
        batch.max_action_num = max([len(ex.tgt_action) for ex in ex_list])
        batch.question_unk_mask, batch.table_unk_mask, batch.column_unk_mask = None, None, None
    else:
        word2vec, unk_idx = Example.word2vec, Example.word_vocab[UNK]
        batch.question_unk_mask, batch.table_unk_mask, batch.column_unk_mask = None, None, None
        question_unk_mask = (batch.questions == unk_idx).cpu()
        if question_unk_mask.any().item():
            raw_questions = np.array([ex.question + [PAD] * (batch.max_question_len - len(ex.question)) for ex in ex_list], dtype='<U100')
            unk_words = raw_questions[question_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.question_unk_mask = question_unk_mask.masked_scatter_(torch.clone(question_unk_mask), oov_flag).to(device)
                batch.question_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)

        table_unk_mask = (batch.tables == unk_idx).cpu()
        if table_unk_mask.any().item():
            raw_tables = np.array([t + [PAD] * (batch.max_table_word_len - len(t)) for e_id, ex in enumerate(ex_list) for t_id, t in enumerate(ex.table) if t_id in select_tables[e_id]], dtype='<U100')
            unk_words = raw_tables[table_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.table_unk_mask = table_unk_mask.masked_scatter_(torch.clone(table_unk_mask), oov_flag).to(device)
                batch.table_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)

        column_unk_mask = (batch.columns == unk_idx).cpu()
        if column_unk_mask.any().item():
            raw_columns = np.array([c + [PAD] * (batch.max_column_word_len - len(c)) for e_id, ex in enumerate(ex_list) for c_id, c in enumerate(ex.column) if c_id in select_columns[e_id]], dtype='<U100')
            unk_words = raw_columns[column_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.column_unk_mask = column_unk_mask.masked_scatter_(torch.clone(column_unk_mask), oov_flag).to(device)
                batch.column_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)

    raw_table_lens = [len(ex.table) for ex in ex_list]
    select_relations = [
        list(range(question_lens[e_id])) + [question_lens[e_id] + tab_id for tab_id in select_tables[e_id]] +
        [question_lens[e_id] + raw_table_lens[e_id] + col_id for col_id in select_columns[e_id]]
        for e_id in range(len(batch))
    ]
    relation_ids = [
        ex.relation_id[:, select_relations[e_id]][select_relations[e_id], :]
        for e_id, ex in enumerate(ex_list)
    ]
    pad_idx = Example.relative_position_vocab['no-relation']
    relations = [ F.pad(torch.tensor(relation_id, dtype=torch.long),
        (0, batch.max_len - relation_id.shape[0], 0, batch.max_len - relation_id.shape[0]),
        value=pad_idx) for relation_id in relation_ids ]
    batch.relations = torch.stack(relations, dim=0).to(device)
    batch.relations_mask = (batch.relations == pad_idx)
    return batch

def from_example_list_ratsql_golden(ex_list, device='cpu', train=True, schema='full-full', **kwargs):
    """ Training uses original schema, but during evaluation, use golden schema (both encoding and decoding phase)
    """
    assert schema in ['full-full', 'full-golden', 'golden-golden']
    if schema == 'full-full': # use full schema for encoding and decoding as usual
        return from_example_list_ratsql(ex_list, device=device, train=train, **kwargs)
    elif schema == 'full-golden': # use full schema as input graph, but after encoding, select golden schema items for decoding
        return from_example_list_ratsql_golden_decode(ex_list, device=device, train=train)
    else: # use golden schema for both encoding and decoding
        return from_example_list_ratsql_golden_schema(ex_list, device=device, train=train)

def from_example_list_graph_pruning(ex_list, device='cpu', train=True, loss_function='bce', **kargs):
    batch = from_example_list_ratsql(ex_list, device=device, train=train, **kargs)
    select_tables = [ex.used_tables for ex in ex_list]
    select_columns = [ex.used_columns for ex in ex_list]
    table_labels = torch.zeros((len(batch), batch.max_table_len), dtype=torch.bool)
    for e_id in range(len(batch)):
        for t_id in select_tables[e_id]:
            table_labels[e_id, t_id] = True
    column_labels = torch.zeros((len(batch), batch.max_column_len), dtype=torch.bool)
    for e_id in range(len(batch)):
        for c_id in select_columns[e_id]:
            column_labels[e_id, c_id] = True
    select_mask = torch.cat([table_labels, column_labels], dim=1)
    batch.select_mask = select_mask.to(device)
    if train:
        # need prune_labels
        mask = torch.cat([batch.table_mask, batch.column_mask], dim=1)
        prune_labels = batch.select_mask.masked_select(mask)
        smoothing = kargs['ls']
        if smoothing > 0 :
            batch.prune_labels = prune_labels.float().masked_fill_(~ prune_labels, 2 * smoothing) - smoothing
        else:
            prune_labels = torch.cat([torch.from_numpy(ex.ls) for ex in ex_list], dim=0).float()
            batch.prune_labels = prune_labels.to(device)
    # provide corresponding table_ids for each column
    column2table_ids = [list(map(lambda x: x[0], ex.db['column_names'][1:])) + [-1e8] * (batch.max_column_len - len(ex.db['column_names'])) for ex in ex_list]
    column2table_ids = torch.tensor(column2table_ids, dtype=torch.long, device=device)
    bias = torch.arange(len(batch), dtype=torch.long, device=device) * batch.max_table_len
    column2table_ids += bias.unsqueeze(-1)
    batch.column2table_ids = column2table_ids
    batch.threshold = 0.5
    return batch

def from_example_list_ratsql_coarse2fine(ex_list, device='cpu', train=True, **kargs):
    batch = from_example_list_graph_pruning(ex_list, device=device, train=train, **kargs)
    if train:
        min_rate, max_rate = kargs.pop('min_rate', 0.05), kargs.pop('max_rate', 1.0)
        sampling_rate = [ex.table_sampling_prob + [0.] * (batch.max_table_len - len(ex.table)) + ex.column_sampling_prob + [0.] * (batch.max_column_len - len(ex.column)) for ex in ex_list]
        sampling_rate = torch.tensor(sampling_rate, dtype=torch.float, device=device).clamp(min_rate, max_rate)
        select_mask_noisy = torch.bernoulli(sampling_rate).bool().masked_fill_(~ torch.cat([batch.table_mask, batch.column_mask], dim=1), False)
        batch.select_mask_noisy = select_mask_noisy | batch.select_mask
    # batch.threshold = 0.5 if train else 0.25
    return batch

class Batch():

    max_q, max_t, max_c = 50, 50, 200
    #generic_relation = obtain_generic_relation(max_q, max_t, max_c)

    def __init__(self, examples, device='cpu'):
        super(Batch, self).__init__()
        self.examples = examples
        self.device = device

    @classmethod
    def from_example_list(cls, ex_list, device='cpu', train=True, method='ratsql', **kwargs):
        method_dict = {
            "ratsql": from_example_list_ratsql,
            "ratsql_golden": from_example_list_ratsql_golden,
            "ratsql_coarse2fine": from_example_list_ratsql_coarse2fine,
            "graph_pruning": from_example_list_graph_pruning
        }
        return method_dict[method](ex_list, device, train=train, **kwargs)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @cached_property
    def lens(self):
        return self.question_lens + self.table_lens + self.column_lens

    @cached_property
    def max_len(self):
        return torch.max(self.lens).item()

    @cached_property
    def max_question_len(self):
        return torch.max(self.question_lens).item()

    @cached_property
    def max_table_len(self):
        return torch.max(self.table_lens).item()

    @cached_property
    def max_column_len(self):
        return torch.max(self.column_lens).item()

    @cached_property
    def max_table_word_len(self):
        return torch.max(self.table_word_lens).item()

    @cached_property
    def max_column_word_len(self):
        return torch.max(self.column_word_lens).item()

    @cached_property
    def max_question_subword_len(self):
        return torch.max(self.question_subword_lens).item()

    @cached_property
    def max_table_subword_len(self):
        return torch.max(self.table_subword_lens).item()

    @cached_property
    def max_column_subword_len(self):
        return torch.max(self.column_subword_lens).item()

    @cached_property
    def golden_lens(self):
        return self.question_lens + self.golden_table_lens + self.golden_column_lens

    @cached_property
    def max_golden_len(self):
        return torch.max(self.golden_lens).item()

    @cached_property
    def max_golden_table_len(self):
        return torch.max(self.golden_table_lens).item()

    @cached_property
    def max_golden_column_len(self):
        return torch.max(self.golden_column_lens).item()

    @cached_property
    def question_mask(self):
        return lens2mask(self.question_lens)

    @cached_property
    def table_mask(self):
        return lens2mask(self.table_lens)

    @cached_property
    def column_mask(self):
        return lens2mask(self.column_lens)

    @cached_property
    def table_word_mask(self):
        return lens2mask(self.table_word_lens)

    @cached_property
    def column_word_mask(self):
        return lens2mask(self.column_word_lens)

    @cached_property
    def question_subword_mask(self):
        return lens2mask(self.question_subword_lens)

    @cached_property
    def table_subword_mask(self):
        return lens2mask(self.table_subword_lens)

    @cached_property
    def column_subword_mask(self):
        return lens2mask(self.column_subword_lens)

    @cached_property
    def mask(self):
        return lens2mask(self.lens)

    """ split means different types of nodes are seperated instead of concatenated together """
    @cached_property
    def mask_split(self):
        return torch.cat([self.question_mask, self.table_mask, self.column_mask], dim=1)

    """ golden means that we only consider relevant schema items """
    @cached_property
    def golden_table_mask(self):
        return lens2mask(self.golden_table_lens)

    @cached_property
    def golden_column_mask(self):
        return lens2mask(self.golden_column_lens)

    @cached_property
    def golden_mask(self):
        return lens2mask(self.golden_lens)

    @cached_property
    def golden_mask_split(self):
        return torch.cat([self.question_mask, self.golden_table_mask, self.golden_column_mask], dim=1)

    def get_frontier_field_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.field2id[e.tgt_action[t].frontier_field])
                # assert self.grammar.id2field[ids[-1]] == e.tgt_action[t].frontier_field
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_frontier_prod_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.prod2id[e.tgt_action[t].frontier_prod])
                # assert self.grammar.id2prod[ids[-1]] == e.tgt_action[t].frontier_prod
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_frontier_field_type_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.type2id[e.tgt_action[t].frontier_field.type])
                # assert self.grammar.id2type[ids[-1]] == e.tgt_action[t].frontier_field.type
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)
