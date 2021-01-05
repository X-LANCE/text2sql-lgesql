#coding=utf8
import torch
import torch.nn as nn
from model.model_utils import Registrable, lens2mask
from model.encoder.graph_encoder import *
from model.encoder.graph_output import *
from model.decoder.sql_parser import *

@Registrable.register('ratsql')
class RATSQL(nn.Module):
    def __init__(self, args, transition_system):
        super(RATSQL, self).__init__()
        self.encoder = Registrable.by_name('encoder_ratsql')(args)
        self.encoder2decoder = PoolingFunction(args.gnn_hidden_size, args.lstm_hidden_size, method='attentive-pooling')
        self.decoder = Registrable.by_name('decoder_tranx')(args, transition_system)

    def forward(self, batch):
        """ This function is used during training, which returns the entire training loss
        """
        encodings, mask = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        loss = self.decoder.score(encodings, mask, h0, batch, golden=False)
        return loss

    def parse(self, batch, beam_size):
        """ This function is used for decoding, which returns a batch of [DecodeHypothesis()] * beam_size
        """
        encodings, mask = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        hyps = []
        for i in range(len(batch)):
            """ table_mappings and column_mappings are used to map original database ids to local ids,
            while reverse_mappings perform the opposite function, mapping local ids to global ids
            """
            hyps.append(self.decoder.parse(encodings[i:i+1], mask[i:i+1], h0[i:i+1], batch, beam_size,
            golden=False, table_mapping=batch.table_mappings[i], column_mapping=batch.column_mappings[i],
            table_reverse_mapping=batch.table_reverse_mappings[i], column_reverse_mapping=batch.column_reverse_mappings[i]))
        return hyps

    def pad_embedding_grad_zero(self, index=None):
        """ Gradients for <pad> symbol and no-relation is always 0;
        Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.encoder.input_layer.pad_embedding_grad_zero(index)
        self.encoder.hidden_layer.pad_embedding_grad_zero()

@Registrable.register('ratsql_golden')
class RATSQLGolden(RATSQL):
    """ Different combinations of full/golden schema on encoder and decoder side, e.g.
    a. encoder full schema, decoder full schema: this is the original ratsql model
    b. encoder full schema, decoder golden schema: after encoding, need to select the indexes from encoded schema items for later usage,
        such as decoder init, attention calculation and select table/column
    c. encoder golden schema, decoder golden schema: clean schema graph, where only the related schema items are provided
    For training and dev dataset, we could use different choices above. But remember to provide additional fields in Batch obj if pruning is needed~(case b.):
        batch.golden_table_lens, batch.golden_column_lens, batch.golden_index
    """
    def forward(self, batch):
        if hasattr(batch, 'golden_table_lens'):
            coarse_encodings, _ = self.encoder(batch)
            source = coarse_encodings.contiguous().view(-1, coarse_encodings.size(-1)).index_select(dim=0, index=batch.golden_index)
            encodings = coarse_encodings.new_zeros(len(batch), batch.golden_mask_split.size(1), coarse_encodings.size(-1))
            encodings = encodings.masked_scatter_(batch.golden_mask_split.unsqueeze(-1), source)
            mask = batch.golden_mask_split
        else:
            encodings, mask = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        loss = self.decoder.score(encodings, mask, h0, batch, golden=hasattr(batch, 'golden_table_lens'))
        return loss

    def parse(self, batch, beam_size):
        if hasattr(batch, 'golden_table_lens'):
            coarse_encodings, _ = self.encoder(batch)
            source = coarse_encodings.contiguous().view(-1, coarse_encodings.size(-1)).index_select(dim=0, index=batch.golden_index)
            encodings = coarse_encodings.new_zeros(len(batch), batch.golden_mask_split.size(1), coarse_encodings.size(-1))
            encodings = encodings.masked_scatter_(batch.golden_mask_split.unsqueeze(-1), source)
            mask = batch.golden_mask_split
        else:
            encodings, mask = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        hyps = []
        for i in range(len(batch)):
            hyps.append(self.decoder.parse(encodings[i:i+1], mask[i:i+1], h0[i:i+1], batch, beam_size,
            golden=hasattr(batch, 'golden_table_lens'), table_mapping=batch.table_mappings[i], column_mapping=batch.column_mappings[i],
            table_reverse_mapping=batch.table_reverse_mappings[i], column_reverse_mapping=batch.column_reverse_mappings[i]))
        return hyps

@Registrable.register('ratsql_coarse2fine')
class RATSQLCoarse2Fine(nn.Module):
    """ Different from RATSQLGolden, ground truth golden schema items are not provided during evaluation,
    we need to predict or classify each schema item.
    """
    def __init__(self,  args, transition_system):
        super(RATSQLCoarse2Fine, self).__init__()
        self.args = args
        self.encoder = Registrable.by_name('encoder_coarse2fine')(args)
        self.encoder2decoder = PoolingFunction(args.gnn_hidden_size, args.lstm_hidden_size, method='attentive-pooling')
        self.decoder = Registrable.by_name('decoder_tranx')(args, transition_system)

    def forward(self, batch, mode='text2sql'):
        assert mode in ['graph_pruning', 'graph_matching', 'text2sql', 'multitask']
        if mode == 'graph_pruning' or mode == 'graph_matching':
            return self.encoder(batch, mode=mode)
        if mode == 'text2sql':
            encodings, mask = self.encoder(batch, mode=mode)
            h0 = self.encoder2decoder(encodings, mask=mask)
            loss = self.decoder.score(encodings, mask, h0, batch, golden=self.args.prune)
            return loss
        else: # multitasking: graph_pruning + text2sql
            encodings, mask, prune_loss = self.encoder(batch, mode=mode)
            h0 = self.encoder2decoder(encodings, mask=mask)
            loss = self.decoder.score(encodings, mask, h0, batch, golden=self.args.prune)
            return loss, prune_loss

    def parse(self, batch, beam_size=5, mode='text2sql'):
        """ select_mask is bool tensor of size bsize x [max(table_lens) + max(column_lens)], indicating which schema item is picked
        """
        assert mode in ['graph_pruning', 'graph_matching', 'text2sql', 'multitask']
        if mode == 'graph_pruning' or mode == 'graph_matching':
            return self.encoder(batch, mode=mode)
        if mode == 'text2sql':
            encodings, mask = self.encoder(batch, mode=mode)
        else: # multitasking: graph_pruning + text2sql
            encodings, mask, select_mask = self.encoder(batch, mode=mode)
        h0 = self.encoder2decoder(encodings, mask=mask)
        hyps = []
        for i in range(len(batch)):
            """ table/column mapping is used to map real database id to temporary positions in encodings
            table/column reverse mappings is used to convert positions in encodings to real database ids
            """
            hyps.append(self.decoder.parse(encodings[i:i+1], mask[i:i+1], h0[i:i+1], batch, beam_size,
                golden=self.args.prune, table_mapping=batch.table_mappings[i], column_mapping=batch.column_mappings[i],
                table_reverse_mapping=batch.table_reverse_mappings[i], column_reverse_mapping=batch.column_reverse_mappings[i]))
        if mode == 'text2sql':
            return hyps
        else:
            return hyps, select_mask

    def pad_embedding_grad_zero(self, index=None):
        """ Gradients for <pad> symbol and no-relation is always 0;
        Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.encoder.input_layer.pad_embedding_grad_zero(index)
        if self.encoder.shared_num_layers > 0:
            self.encoder.hidden_layer.pad_embedding_grad_zero()
        if self.args.shared_num_layers < self.args.gnn_num_layers:
            self.encoder.output_layer.private_sql_encoder.pad_embedding_grad_zero()
            self.encoder.output_layer.private_aux_encoder.pad_embedding_grad_zero()
