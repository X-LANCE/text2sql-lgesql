#coding=utf8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import lens2mask, tile

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
            self.attn = MultiHeadAttention(hidden_size, hidden_size, hidden_size, bias=bias, dropout=dropout, heads=heads)
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
    def __init__(self, hidden_size, query_size, key_value_size, bias=True, dropout=0., heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = int(heads)
        self.hidden_size = hidden_size
        assert self.hidden_size % self.heads == 0, 'Head num %d must be divided by hidden size %d' % (heads, hidden_size)
        self.d_k = self.hidden_size // self.heads
        self.dropout_layer = nn.Dropout(p=dropout)
        self.W_q, self.W_k, self.W_v = nn.Linear(query_size, self.hidden_size, bias=bias), \
                nn.Linear(key_value_size, self.hidden_size, bias=False), nn.Linear(key_value_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(self.hidden_size, query_size, bias=bias)

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

    def forward(self, logits, mask, labels):
        selected_logits = logits.masked_select(mask)
        loss =  self.function(selected_logits, labels)
        return loss

class GraphOutputLayer(nn.Module):

    def __init__(self, hidden_size):
        super(GraphOutputLayer, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, inputs, batch):
        """ Re-scatter data format:
                inputs: sum(q_len + t_len + c_len) x hidden_size
                outputs: bsize x (max_q_len + max_t_len + max_c_len) x hidden_size
        """
        outputs = inputs.new_zeros(len(batch), batch.mask.size(1), self.hidden_size)
        outputs = outputs.masked_scatter_(batch.mask.unsqueeze(-1), inputs)
        return outputs, batch.mask

class GraphOutputLayerWithPruning(nn.Module):

    def __init__(self, hidden_size):
        super(GraphOutputLayerWithPruning, self).__init__()
    
    def forward(self, inputs, batch):
        outputs = inputs.new_zeros(len(batch), batch.mask.size(1), self.hidden_size)
        outputs = outputs.masked_scatter_(batch.mask.unsqueeze(-1), inputs)
        q_outputs, s_outputs = outputs[:, :batch.max_question_len], outputs[:, batch.max_question_len:]


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
