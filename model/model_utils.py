#coding=utf8
import copy
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def lens2mask(lens):
    bsize = lens.numel()
    max_len = lens.max()
    masks = torch.arange(0, max_len).type_as(lens).to(lens.device).repeat(bsize, 1).lt(lens.unsqueeze(1))
    masks.requires_grad = False
    return masks

def mask2matrix(mask):
    col_mask, row_mask = mask.unsqueeze(-1), mask.unsqueeze(-2)
    return col_mask & row_mask

def tile(x, count, dim=0):
    """
        Tiles x on dimension dim count times.
        E.g. [1, 2, 3], count=2 ==> [1, 1, 2, 2, 3, 3]
            [[1, 2], [3, 4]], count=3, dim=1 ==> [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]
        Different from torch.repeat
    """
    if x is None:
        return x
    elif type(x) in [list, tuple]:
        return type(x)([tile(each, count, dim) for each in x])
    else:
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.contiguous().view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x

def rnn_wrapper(encoder, inputs, lens, cell='lstm'):
    """
        @args:
            encoder(nn.Module): rnn series bidirectional encoder, batch_first=True
            inputs(torch.FloatTensor): rnn inputs, [bsize x max_seq_len x in_dim]
            lens(torch.LongTensor): seq len for each sample, allow length=0, padding with 0-vector, [bsize]
        @return:
            out(torch.FloatTensor): output of encoder, bsize x max_seq_len x hidden_dim*2
            hidden_states([tuple of ]torch.FloatTensor): final hidden states, num_layers*2 x bsize x hidden_dim
    """
    # rerank according to lens and remove empty inputs
    sorted_lens, sort_key = torch.sort(lens, descending=True)
    nonzero_num, total_num = torch.sum(sorted_lens > 0).item(), sorted_lens.size(0)
    sort_key = sort_key[:nonzero_num]
    sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key)
    # forward non empty inputs    
    packed_inputs = rnn_utils.pack_padded_sequence(sorted_inputs, sorted_lens[:nonzero_num].tolist(), batch_first=True)
    packed_out, sorted_h = encoder(packed_inputs)  # bsize x srclen x dim
    sorted_out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
    if cell.upper() == 'LSTM':
        sorted_h, sorted_c = sorted_h
    # rerank according to sort_key
    out_shape = list(sorted_out.size())
    out_shape[0] = total_num
    out = sorted_out.new_zeros(*out_shape).scatter_(0, sort_key.unsqueeze(-1).unsqueeze(-1).repeat(1, *out_shape[1:]), sorted_out)
    h_shape = list(sorted_h.size())
    h_shape[1] = total_num
    h = sorted_h.new_zeros(*h_shape).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).repeat(h_shape[0], 1, h_shape[-1]), sorted_h)
    if cell.upper() == 'LSTM':
        c = sorted_c.new_zeros(*h_shape).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).repeat(h_shape[0], 1, h_shape[-1]), sorted_c)
        return out, (h.contiguous(), c.contiguous())
    return out, h.contiguous()

class Registrable(object):
    """
    A class that collects all registered components,
    adapted from `common.registrable.Registrable` from AllenNLP
    """
    registered_components = dict()

    @staticmethod
    def register(name):
        def register_class(cls):
            if name in Registrable.registered_components:
                raise RuntimeError('class %s already registered' % name)

            Registrable.registered_components[name] = cls
            return cls

        return register_class

    @staticmethod
    def by_name(name):
        return Registrable.registered_components[name]

class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
