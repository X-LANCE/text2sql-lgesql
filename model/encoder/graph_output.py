#coding=utf8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import lens2mask, mask2matrix, tile
from model.encoder.graph_hidden import RATSQLGraphHiddenLayer as EncodingModule

class PoolingFunction(nn.Module):
    """ Map a sequence of hidden_size dim vectors into one fixed size vector with dimension output_size,
    and expand multiple times if necessary
    """
    def __init__(self, hidden_size=256, output_size=256, dropout=0., heads=8, bias=True, method='multihead-attention'):
        super(PoolingFunction, self).__init__()
        assert method in ['mean-pooling', 'max-pooling', 'attentive-pooling', 'multihead-attention']
        self.method = method
        if self.method == 'attentive-pooling':
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=bias),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=bias)
            )
        elif self.method == 'multihead-attention':
            self.attn = MultiHeadAttention(hidden_size, bias=bias, dropout=dropout, heads=heads)
        self.mapping_function = nn.Sequential(nn.Linear(hidden_size, output_size, bias=bias), nn.Tanh()) \
            if hidden_size != output_size else lambda x: x

    def forward(self, inputs, targets=None, mask=None):
        """
        @args:
            inputs(torch.FloatTensor): features, batch_size x seq_len x hidden_size
            targets(torch.FloatTensor): features, batch_size x tgt_len x hidden_size
            mask(torch.BoolTensor): mask for inputs, batch_size x seq_len
        @return:
            outputs(torch.FloatTensor): aggregate the seq_len dim for inputs
                and expand to tgt_len (targets is not None), batch_size [x tgt_len ]x hidden_size
        """
        if self.method == 'multihead-attention':
            assert targets is not None
            outputs = self.attn(inputs, targets, mask)
            return self.mapping_function(outputs)
        else:
            if self.method == 'max-pooling':
                outputs = inputs.masked_fill(~ mask.unsqueeze(-1), -1e8)
                outputs = outputs.max(dim=1)[0]
            elif self.method == 'mean-pooling':
                mask_float = mask.float().unsqueeze(-1)
                outputs = (inputs * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            elif self.method == 'attentive-pooling':
                e = self.attn(inputs).squeeze(-1)
                e = e + (1 - mask.float()) * (-1e20)
                a = torch.softmax(e, dim=1).unsqueeze(1)
                outputs = torch.bmm(a, inputs).squeeze(1)
            else:
                raise ValueError('[Error]: Unrecognized pooling method %s !' % (self.method))
            outputs = self.mapping_function(outputs)
            if targets is not None:
                return outputs.unsqueeze(1).expand(-1, targets.size(1), -1)
            else:
                return outputs

class MultiHeadAttention(nn.Module):
    """ Transformer scaled dot production module
        head(Q,K,V) = softmax(QW_qKW_k^T / sqrt(d_k)) VW_v
        MultiHead(Q,K,V) = Concat(head_1,head_2,...,head_n) W_o
    """
    def __init__(self, hidden_size, bias=True, dropout=0., heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = int(heads)
        self.hidden_size = hidden_size
        assert self.hidden_size % self.heads == 0, 'Head num %d must be divided by hidden size %d' % (heads, hidden_size)
        self.d_k = self.hidden_size // self.heads
        self.dropout_layer = nn.Dropout(p=dropout)
        self.W_q, self.W_k, self.W_v = nn.Linear(self.hidden_size, self.hidden_size, bias=bias), \
                nn.Linear(self.hidden_size, self.hidden_size, bias=bias), nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

    def forward(self, hiddens, query_hiddens, mask=None):
        '''
            @params:
                hiddens : encoded sequence representations, bsize x seqlen x hidden_size
                query_hiddens : bsize x tgtlen x hidden_size
                mask : length mask for hiddens, ByteTensor, bsize x seqlen
            @return:
                context : bsize x tgtlen x hidden_size
        '''
        Q, K, V = self.W_q(self.dropout_layer(query_hiddens)), self.W_k(self.dropout_layer(hiddens)), self.W_v(self.dropout_layer(hiddens))
        Q, K, V = Q.reshape(-1, Q.size(1), 1, self.heads, self.d_k), K.reshape(-1, 1, K.size(1), self.heads, self.d_k), V.reshape(-1, 1, V.size(1), self.heads, self.d_k)
        e = (Q * K).sum(-1) / math.sqrt(self.d_k) # bsize x tgtlen x seqlen x heads
        if mask is not None:
            e = e + ((1 - mask.float()) * (-1e20)).unsqueeze(1).unsqueeze(-1)
        a = torch.softmax(e, dim=2)
        concat = (a.unsqueeze(-1) * V).sum(dim=2).reshape(-1, query_hiddens.size(1), self.hidden_size)
        context = self.W_o(concat)
        return context

class ScoreFunction(nn.Module):
    def __init__(self, hidden_size, mlp=1, method='biaffine'):
        super(ScoreFunction, self).__init__()
        assert method in ['dot', 'bilinear', 'affine', 'biaffine']
        self.mlp = int(mlp)
        self.hidden_size = hidden_size // self.mlp
        if self.mlp > 1: # use mlp to perform dim reduction
            self.mlp_q = nn.Sequential(nn.Linear(hidden_size, self.hidden_size), nn.Tanh())
            self.mlp_s = nn.Sequential(nn.Linear(hidden_size, self.hidden_size), nn.Tanh())
        self.method = method
        if self.method == 'bilinear':
            self.W = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif self.method == 'affine':
            self.affine = nn.Linear(self.hidden_size * 2, 1)
        elif self.method == 'biaffine':
            self.W = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.affine = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, question, schema):
        """
        @args:
            question(torch.FloatTensor): bsize x schema_len x hidden_size
            schema(torch.FloatTensor): bsize x schema_len x hidden_size
        @return:
            scores(torch.FloatTensor): bsize x schema_len
        """
        if self.mlp > 1:
            question, schema = self.mlp_q(question), self.mlp_s(schema)
        if self.method == 'dot':
            scores = (question * schema).sum(dim=-1)
        elif self.method == 'bilinear':
            scores = (question * self.W(schema)).sum(dim=-1)
        elif self.method == 'affine':
            scores = self.affine(torch.cat([question, schema], dim=-1)).squeeze(-1)
        elif self.method == 'biaffine':
            scores = (question * self.W(schema)).sum(dim=-1)
            scores += self.affine(torch.cat([question, schema], dim=-1)).squeeze(-1)
        else:
            raise ValueError('[Error]: Unrecognized score function method %s!' % (self.method))
        return scores

class LossFunction(nn.Module):
    def __init__(self, method='bce', alpha=0.5, gamma=2, pos_weight=1.0, reduction='sum'):
        super(LossFunction, self).__init__()
        self.method = method
        if self.method == 'bce':
            self.function = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=torch.full([1], pos_weight))
        elif self.method == 'focal':
            self.function = BinaryFocalLoss(gamma, alpha=alpha, reduction=reduction)
        else:
            raise ValueError('[Error]: Unrecognized loss function method %s !' % (self.method))

    def forward(self, logits, mask, batch):
        selected_logits = logits.masked_select(mask)
        loss =  self.function(selected_logits, batch.prune_labels)
        return loss

class RATSQLGraphOutputLayer(nn.Module):

    def __init__(self, enc_dim):
        super(RATSQLGraphOutputLayer, self).__init__()
        self.enc_dim = enc_dim

    def forward(self, inputs, batch):
        """
            inputs: bsize x max(q_len + t_len + c_len) x enc_dim
            outputs: bsize x (max_q_len + max_t_len + max_c_len) x enc_dim
        """
        mask, mask_split = batch.mask, batch.mask_split
        source = inputs.masked_select(mask.unsqueeze(-1))
        outputs = source.new_zeros(len(batch), mask_split.size(1), self.enc_dim)
        outputs = outputs.masked_scatter_(mask_split.unsqueeze(-1), source)
        return outputs, mask_split

class Coarse2FineGraphOutputLayer(nn.Module):
    """ Classifying each schema node into relevant or irrelevant node, and check whether the given schema graph can answer the given question,
    optionally extract the subgraph from the entire database schema graph
        prune: if True, remove irrelevant schema items for decoding
    """
    def __init__(self, args):
        super(Coarse2FineGraphOutputLayer, self).__init__()
        self.hidden_size = args.gnn_hidden_size
        self.transform_output = RATSQLGraphOutputLayer(args.gnn_hidden_size)
        self.private_num_layers = args.gnn_num_layers - args.shared_num_layers
        self.graph_matching = GraphMatching(args.gnn_hidden_size) # TODO:
        self.graph_pruning = GraphPruning(args.gnn_hidden_size, args.dropout, args.num_heads,
            question_pooling_method=args.question_pooling_method, score_function=args.score_function, mlp=args.dim_reduction,
            loss_function=args.loss_function, alpha=args.alpha, gamma=args.gamma, pos_weight=args.pos_weight)
        self.prune = args.prune
        self.private_sql_encoder, self.private_aux_encoder = None, None
        if self.private_num_layers > 0:
            self.private_sql_encoder = EncodingModule(args.gnn_hidden_size, args.relation_num, args.num_heads,
                num_layers=self.private_num_layers, feat_drop=args.dropout, attn_drop=args.attn_drop,
                share_layers=True, share_heads=True, layerwise_relation=False)
            self.private_aux_encoder = EncodingModule(args.gnn_hidden_size, args.relation_num, args.num_heads,
                num_layers=self.private_num_layers, feat_drop=args.dropout, attn_drop=args.attn_drop,
                share_layers=True, share_heads=True, layerwise_relation=False)

    def forward(self, inputs, batch, mode='text2sql'):
        """ Given used_tables and used_columns fields in batch, extract subgraph schema
        @args:
            inputs: bsize x max(question_len + table_len + column_len) x hidden_size
            batch: we use fields select_mask, prune_labels, matching_labels
        @return:
            outputs: if prune: bsize x [max(question_lens) + max(golden_table_lens) + max(golden_column_lens)] x hidden_size,
                o.w. bsize x [max(question_lens) + max(table_lens) + max(column_lens)] x hidden_size
            aux_outputs: auxiliary loss, including graph matching loss or graph pruning loss if training, o.w. schema matching label or schema select mask
        """
        # auxiliary task: graph pruning and graph matching computing
        aux_outputs = self.private_aux_encoder(inputs, batch.relations, batch.relations_mask) if self.private_aux_encoder is not None else inputs
        inputs_t, mask_t = self.transform_output(aux_outputs, batch)
        if mode == 'graph_matching':
            aux_outputs, select_mask = None, None
        elif mode == 'graph_pruning':
            aux_outputs, select_mask = self.graph_pruning(inputs_t, mask_t, batch, noisy=False, constrain='column->table')
        elif mode == 'multitask' or (self.prune and mode == 'text2sql'):
            aux_outputs, select_mask = self.graph_pruning(inputs_t, mask_t, batch, noisy=True, constrain='column->table')
        else:
            aux_outputs, select_mask = None, None

        if mode == 'graph_matching' or mode == 'graph_pruning':
            aux_outputs = aux_outputs if self.training else select_mask
            return aux_outputs # loss or predictions

        # main task: text2sql encoder
        if self.private_sql_encoder is not None:
            sql_outputs = self.private_sql_encoder(inputs, batch.relations, batch.relations_mask)
            inputs_t, mask_t = self.transform_output(sql_outputs, batch)
        if self.prune: # use select_mask to filter irrelevant schema items for decoding
            retain_mask = torch.cat([batch.question_mask, select_mask], dim=1)
            batch = self.prepare_pruning_fields(retain_mask, batch)
            source = inputs_t.masked_select(retain_mask.unsqueeze(-1))
            outputs = inputs_t.new_zeros(len(batch), batch.golden_mask_split.size(1), self.hidden_size)
            outputs = outputs.masked_scatter_(batch.golden_mask_split.unsqueeze(-1), source)
            mask = batch.golden_mask_split
        else:
            outputs, mask = inputs_t, mask_t
        if mode == 'text2sql':
            return outputs, mask
        else:
            aux_outputs = aux_outputs if self.training else select_mask
            return outputs, mask, aux_outputs

    def prepare_pruning_fields(self, retain_mask, batch):
        # re-calculate relevant table and column lens
        table_select_mask = retain_mask[:, batch.max_question_len: batch.max_question_len + batch.max_table_len]
        column_select_mask = retain_mask[:, batch.max_question_len + batch.max_table_len:]
        batch.golden_table_lens = table_select_mask.int().sum(dim=1)
        batch.golden_column_lens = column_select_mask.int().sum(dim=1)
        # re-define mappings and reverse_mappings
        select_table_index = torch.arange(len(batch) * batch.max_table_len, dtype=torch.long, device=retain_mask.device)[table_select_mask.contiguous().view(-1)]
        e_ids = select_table_index // batch.max_table_len
        t_ids = select_table_index - e_ids * batch.max_table_len
        table_reverse_mappings, table_mappings = [[] for _ in range(len(batch))], [dict() for _ in range(len(batch))]
        for e_id, t_id in zip(e_ids.tolist(), t_ids.tolist()):
            global_id = batch.table_reverse_mappings[e_id][t_id]
            table_mappings[e_id][global_id] = len(table_reverse_mappings[e_id])
            table_reverse_mappings[e_id].append(global_id)
        select_column_index = torch.arange(len(batch) * batch.max_column_len, dtype=torch.long, device=retain_mask.device)[column_select_mask.contiguous().view(-1)]
        e_ids = select_column_index // batch.max_column_len
        c_ids = select_column_index - e_ids * batch.max_column_len
        column_reverse_mappings, column_mappings = [[] for _ in range(len(batch))], [dict() for _ in range(len(batch))]
        for e_id, c_id in zip(e_ids.tolist(), c_ids.tolist()):
            global_id = batch.column_reverse_mappings[e_id][c_id]
            column_mappings[e_id][global_id] = len(column_reverse_mappings[e_id])
            column_reverse_mappings[e_id].append(global_id)
        batch.table_mappings, batch.table_reverse_mappings = table_mappings, table_reverse_mappings
        batch.column_mappings, batch.column_reverse_mappings = column_mappings, column_reverse_mappings
        return batch

class GraphPruning(nn.Module):
    """ Select the relevant schema items from the entire graph by classifying each node
    """
    def __init__(self, hidden_size, dropout=0.2, heads=8, question_pooling_method='multihead-attention', score_function='affine', mlp=2, loss_function='bce', alpha=0.5, gamma=2, pos_weight=1.0):
        super(GraphPruning, self).__init__()
        self.hidden_size = hidden_size
        self.question_pooling = PoolingFunction(hidden_size, hidden_size, heads=heads, method=question_pooling_method)
        self.score_function = ScoreFunction(hidden_size, mlp=mlp, method=score_function)
        self.loss_function = LossFunction(method=loss_function, alpha=alpha, gamma=gamma, pos_weight=pos_weight, reduction='sum')

    def forward(self, inputs, mask, batch, noisy=True, constrain='column->table'):
        """ inputs: bsize x [max(q_len) + max(t_len) + max(c_len)] x hidden_size
            mask: bsize x [max(q_len) + max(t_len) + max(c_len)]
            batch: contain golden labels in field batch.prune_labels, see utils/batch.py
            noisy: during training, whether add noise for the golden schema items for text2sql decoder
        (Done) a. choose one schema constraint:
            table->column constraint: if one table is not selected, all of its columns are not selected
            column->table constraint: if one column is selected, its corresponding table must be selected
        (Done) b. at least one table/column must be selected
        """
        q_inputs, s_inputs = inputs[:, :batch.max_question_len], inputs[:, batch.max_question_len:]
        q_mask, s_mask = mask[:, :batch.max_question_len], mask[:, batch.max_question_len:]
        questions = self.question_pooling(q_inputs, s_inputs, q_mask)
        logits = self.score_function(questions, s_inputs)
        if self.training:
            scores = self.loss_function(logits, s_mask, batch)
            if noisy:
                pred_scores = torch.sigmoid(logits)
                pred_scores = pred_scores.masked_fill_(~ s_mask, -1e8)
                select_mask = (pred_scores >= batch.threshold) | batch.select_mask_noisy
            else:
                select_mask = batch.select_mask
        else:
            scores = torch.sigmoid(logits)
            scores = scores.masked_fill_(~ s_mask, -1e8) # not choose padding symbol
            select_mask = scores >= batch.threshold
            _, max_t_idx = scores[:, :batch.max_table_len].max(dim=1) # at least pick one table
            _, max_c_idx = scores[:, batch.max_table_len:].max(dim=1) # at least pick one column
            max_c_idx += batch.max_table_len
            max_idx = torch.stack([max_t_idx, max_c_idx], dim=1)
            select_mask = select_mask.scatter_(1, max_idx, True)
            # select_mask = batch.select_mask # use golden label to check the upperbound
        if noisy or not self.training:
            table_mask, column_mask = select_mask[:, :batch.max_table_len], select_mask[:, batch.max_table_len:]
            table_ids = batch.column2table_ids.masked_select(column_mask[:, 1:]) # ignore special column *
            if constrain == 'table->column': # table->column constraint
                fill_table_mask = torch.gather(table_mask.contiguous().view(-1), dim=0, index=table_ids)
                column_mask[:, 1:].masked_scatter_(column_mask[:, 1:].clone(), fill_table_mask)
                select_mask = torch.cat([table_mask, column_mask], dim=1)
            elif constrain == 'column->table': # column->table constraint
                table_mask = table_mask.contiguous().view(-1).scatter_(0, table_ids, True)
                select_mask = torch.cat([table_mask.view(len(batch), -1), column_mask], dim=1)
        return scores, select_mask

class GraphMatching(nn.Module):
    """ Check whether the question and the given database schema are matched,
    in other words, whether the question is answerable by the given schema
    """
    def __init__(self, hidden_size):
        super(GraphMatching, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, inputs, mask, batch):
        return None

    def parse(self, inputs, mask, batch):
        return None

class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma, alpha=0., reduction='sum'):
        """ alpha can be viewed as label re-weight factor, such as pos_weight param in BCEWithLogitsLoss
        0 < alpha < 1, alpha * log p_t if label == 1, else (1 - alpha) * log (1 - p_t) if label == 0,
        by default, we do not use this hyper-parameter, None
            gamma, the larger gamma is, the less attention is paid to already correctly classified examples,
        if gamma is 0, equal to traditional BinaryCrossEntropyLoss
        """
        super(BinaryFocalLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.gamma, self.reduction = gamma, reduction
        if alpha >= 1. or alpha <= 0.:
            self.alpha = None
        else:
            self.alpha = torch.tensor([1 - alpha, alpha], dtype=torch.float)

    def forward(self, logits, targets):
        """
            logits: score before sigmoid activation, flattened 1-dim
            targets: label for each logit, FloatTensor, flattened 1-dim
        """
        pt = torch.sigmoid(logits)
        pt = torch.stack([1 - pt, pt], dim=-1) # num x 2
        labels = (targets >= 0.5).long()
        pt = pt.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        logpt = torch.log(pt)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device).gather(0, labels)
            logpt = alpha * logpt
        loss = - (1 - pt)**self.gamma * logpt
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
