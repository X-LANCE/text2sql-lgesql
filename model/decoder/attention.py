#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
        Transformer scaled dot production module
            head(Q,K,V)=softmax(QW_q (KW_k)^T / sqrt(d_k)) VW_v
            MultiHead(Q,K,V)= Concat(head_1,head_2,...,head_n) W_o
    """
    def __init__(self, enc_dim, dec_dim, bias=True, dropout=0., heads=4):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        assert enc_dim % self.heads == 0, 'Head num %d must be divided by encoding dim %d' % (enc_dim, self.heads)
        self.enc_dim, self.dec_dim = enc_dim, dec_dim
        self.d_k = enc_dim // self.heads
        self.dropout_layer = nn.Dropout(p=dropout)
        self.W_q, self.W_k, self.W_v = nn.Linear(self.dec_dim, self.enc_dim, bias=bias), \
                nn.Linear(self.enc_dim, self.enc_dim, bias=bias), nn.Linear(self.enc_dim, self.enc_dim, bias=bias)
        self.W_o = nn.Linear(self.enc_dim, self.enc_dim, bias=bias)

    def forward(self, hiddens, decoder_state, mask=None):
        '''
            @params:
                hiddens : encoded sequence representations, bsize x seqlen x enc_dim
                decoder_state : bsize x dec_dim
                mask : length mask for hiddens, ByteTensor
            @return:
                context : bsize x enc_dim
                a : normalized coefficient, bsize x seqlen
        '''
        Q, K, V = self.W_q(self.dropout_layer(decoder_state)), self.W_k(self.dropout_layer(hiddens)), self.W_v(self.dropout_layer(hiddens))
        Q, K, V = Q.reshape(-1, 1, self.heads, self.d_k), K.reshape(-1, K.size(1), self.heads, self.d_k), V.reshape(-1, V.size(1), self.heads, self.d_k)
        e = (Q * K).sum(-1) / math.sqrt(self.d_k) # bsize x seqlen x heads
        if mask is not None:
            e = e + ((1 - mask.float()) * (-1e20)).unsqueeze(-1)
        a = torch.softmax(e, dim=1)
        concat = (a.unsqueeze(-1) * V).sum(dim=1).reshape(-1, self.enc_dim)
        context = self.W_o(concat)
        return context, a.mean(dim=-1)

class Attention(nn.Module):
    """
        Traditional Seq2Seq attention module
            dot: (h_{dec} W)^T h_{enc}
            nn: v^T tanh( W [ h_{enc} ; h_{dec} ] + bias)
            multi_dimension: feature wise multi dimension softmax
    """
    METHODS = ['dot', 'nn']
    def __init__(self, enc_dim, dec_dim, dropout=0.5, method='nn', bias=False, multi_dimension=False):
        super(Attention, self).__init__()
        self.enc_dim, self.dec_dim = enc_dim, dec_dim
        assert method in Attention.METHODS
        self.method = method
        if self.method == 'dot':
            self.Wa = nn.Linear(self.dec_dim, self.enc_dim, bias=bias)
        else:
            self.Wa = nn.Linear(self.enc_dim + self.dec_dim, self.dec_dim, bias=bias)
            self.Va = nn.Linear(self.dec_dim, 1, bias=bias)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.multi_dimension = multi_dimension
        if self.multi_dimension:
            self.md = nn.Sequential(nn.Linear(self.dec_dim, self.dec_dim), nn.ReLU(), nn.Linear(self.dec_dim, self.dec_dim))

    def forward(self, hiddens, decoder_state, mask=None):
        '''
            @params:
                hiddens : encoded sequence representations, bsize x seqlen x enc_dim
                decoder_state : bsize x dec_dim
                mask : length mask for hiddens, ByteTensor
            @return:
                context : bsize x enc_dim
                a : normalized coefficient, bsize x seqlen if not multi_dimension else bsize x seqlen x dec_dim
        '''
        hiddens, decoder_state = self.dropout_layer(hiddens), self.dropout_layer(decoder_state)
        if self.method == 'dot':
            m = self.Wa(decoder_state)
            e = torch.bmm(m.unsqueeze(1), hiddens.transpose(-1, -2)).squeeze(dim=1)
        elif self.method == 'nn':
            d = decoder_state.unsqueeze(dim=1).repeat(1, hiddens.size(1), 1)
            e = self.Wa(torch.cat([hiddens, d], dim=-1))
            e = self.Va(torch.tanh(e)).squeeze(dim=-1)
        else:
            raise ValueError('Unrecognized attention calculation method: %s' % (self.method))
        if self.multi_dimension:
            e_md = self.md(decoder_state)
            e = e.unsqueeze(dim=-1) + e_md.unsqueeze(dim=1)
            if mask is not None:
                e = e + ((1 - mask.float()) * (-1e20)).unsqueeze(-1)
                # e.masked_fill_(mask.unsqueeze(dim=-1) == 0, -1e20)
            a = F.softmax(e, dim=1)
            context = torch.sum(a * hiddens, dim=1)
        else:
            if mask is not None:
                e = e + (1 - mask.float()) * (-1e20)
                # e.masked_fill_(mask == 0, -1e20)
            a = F.softmax(e, dim=1)
            context = torch.bmm(a.unsqueeze(1), hiddens).squeeze(dim=1)
        return context, a
