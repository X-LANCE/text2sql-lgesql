#coding=utf8
import copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import clones

class RATSQLGraphHiddenLayer(nn.Module):

    def __init__(self, hidden_size, relation_num, num_heads, num_layers=2, feat_drop=0., attn_drop=0., share_layers=True, share_heads=True, layerwise_relation=False):
        super(RATSQLGraphHiddenLayer, self).__init__()
        self.num_layers = num_layers
        self.share_layers, self.share_heads = share_layers, share_heads
        if self.share_layers:
            if self.share_heads:
                self.relation_embed_k = nn.Embedding(relation_num, hidden_size // num_heads, padding_idx=0)
                self.relation_embed_v = nn.Embedding(relation_num, hidden_size // num_heads, padding_idx=0)
            else:
                self.relation_embed_k = nn.Embedding(relation_num, hidden_size, padding_idx=0)
                self.relation_embed_v = nn.Embedding(relation_num, hidden_size, padding_idx=0)
        else:
            self.relation_embed_k, self.relation_embed_v = None, None
        gnn_module = RATSQLGraphIterataionLayer(hidden_size, relation_num, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, share_layers=share_layers, share_heads=share_heads, layerwise_relation=layerwise_relation)
        self.gnn_layers = clones(gnn_module, self.num_layers)

    def pad_embedding_grad_zero(self):
        if self.relation_embed_k is None:
            for i in range(self.num_layers):
                self.gnn_layers[i].relation_embed_k.weight.grad[0].zero_()
                self.gnn_layers[i].relation_embed_v.weight.grad[0].zero_()
        else:
            self.relation_embed_k.weight.grad[0].zero_()
            self.relation_embed_v.weight.grad[0].zero_()

    def forward(self, inputs, relations, relations_mask):
        for i in range(self.num_layers):
            inputs = self.gnn_layers[i](inputs, relations, relations_mask, self.relation_embed_k, self.relation_embed_v)
        return inputs

class RATSQLGraphIterataionLayer(nn.Module):

    def __init__(self, hidden_size, relation_num, num_heads, feedforward=1024, feat_drop=0., attn_drop=0., share_layers=True, share_heads=True, layerwise_relation=False):
        super(RATSQLGraphIterataionLayer, self).__init__()
        assert hidden_size % num_heads == 0, 'Hidden size is not divisible by num of heads'
        self.hidden_size = hidden_size
        self.relation_num = relation_num
        self.num_heads = num_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.concat_affine = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, feedforward),
            # GELU(),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward, self.hidden_size)
        )
        self.layer_norm_1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_size)
        self.share_layers, self.share_heads = share_layers, share_heads
        if not self.share_layers:
            if self.share_heads:
                self.relation_embed_k = nn.Embedding(self.relation_num, self.hidden_size // self.num_heads, padding_idx=0)
                self.relation_embed_v = nn.Embedding(self.relation_num, self.hidden_size // self.num_heads, padding_idx=0)
            else:
                self.relation_embed_k = nn.Embedding(self.relation_num, self.hidden_size, padding_idx=0)
                self.relation_embed_v = nn.Embedding(self.relation_num, self.hidden_size, padding_idx=0)
        else:
            self.relation_embed_k, self.relation_embed_v = None, None
        self.layerwise_relation = layerwise_relation
        if self.layerwise_relation:
            assert self.share_layers, 'If we use layerwise relation, initial edge features must be shared across different layers.'
        input_dim = self.hidden_size // self.num_heads if self.share_heads else self.hidden_size
        self.relation_affine_k = nn.Linear(input_dim, input_dim, bias=False) if self.layerwise_relation else lambda x: x
        self.relation_affine_v = nn.Linear(input_dim, input_dim, bias=False) if self.layerwise_relation else lambda x: x
        self.dropout_layer = nn.Dropout(p=feat_drop)
        self.attndrop_layer = nn.Dropout(p=attn_drop)

    def forward(self, inputs, relations, relations_mask, relation_embed_k=None, relation_embed_v=None):
        bsize, seqlen = inputs.size(0), inputs.size(1)
        # bsize x num_heads x seqlen x 1 x dim
        q = self.query(self.dropout_layer(inputs)).view(bsize, seqlen, self.num_heads, -1).transpose(1, 2).unsqueeze(3)
        # bsize x num_heads x seqlen x seqlen x dim
        k = self.key(self.dropout_layer(inputs)).view(bsize, seqlen, self.num_heads, -1).\
            transpose(1, 2).unsqueeze(2).expand(bsize, self.num_heads, seqlen, seqlen, -1)
        v = self.value(self.dropout_layer(inputs)).view(bsize, seqlen, self.num_heads, -1).\
            transpose(1, 2).unsqueeze(2).expand(bsize, self.num_heads, seqlen, seqlen, -1)
        # bsize x num_heads x seqlen x seqlen x dim
        if not self.share_layers:
            relation_embed_k, relation_embed_v = self.relation_embed_k, self.relation_embed_v
        if self.share_heads:
            relation_k = self.relation_affine_k(relation_embed_k(relations)).unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
            relation_v = self.relation_affine_v(relation_embed_v(relations)).unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
        else:
            relation_k = self.relation_affine_k(relation_embed_k(relations)).view(bsize, seqlen, seqlen, self.num_heads, -1).permute(0, 3, 1, 2, 4)
            relation_v = self.relation_affine_v(relation_embed_v(relations)).view(bsize, seqlen, seqlen, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k + relation_k
        v = v + relation_v
        # e: bsize x heads x seqlen x seqlen
        scale_factor = math.sqrt(self.hidden_size // self.num_heads)
        e = (torch.matmul(q, k.transpose(-1, -2)) / scale_factor).squeeze(-2)
        e = e + (relations_mask.float() * (-1e20)).unsqueeze(1) # mask no-relation
        a = torch.softmax(e, dim=-1)
        a = self.attndrop_layer(a)
        outputs = torch.matmul(a.unsqueeze(-2), v).squeeze(-2)
        outputs = outputs.transpose(1, 2).contiguous().view(bsize, seqlen, -1)
        outputs = self.concat_affine(outputs)
        outputs = self.layer_norm_1(inputs + outputs)
        outputs = self.layer_norm_2(outputs + self.feedforward(outputs))
        return outputs

class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)
