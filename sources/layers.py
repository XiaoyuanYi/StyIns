# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-04 22:20:43
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from made import MADE


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, drop_ratio=0.5):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=True, batch_first=True,
            dropout=(0 if n_layers == 1 else drop_ratio))
        self.dropout_layer = nn.Dropout(drop_ratio)


    def forward(self, ori_embed_seq, input_lens=None):
        # ori_embed_seq: (B, L, emb_dim)
        # input_lens: (B)
        embed_seq = self.dropout_layer(ori_embed_seq)

        if input_lens is None:
            outputs, (state_h, state_c) = self.rnn(embed_seq, None)
        else:
            # Dynamic RNN
            packed = torch.nn.utils.rnn.pack_padded_sequence(embed_seq,
                input_lens, batch_first=True, enforce_sorted=False)
            outputs, (state_h, state_c) = self.rnn(packed, None)
            # outputs: (B, L, 2*H)
            # state: (num_layers*num_directions, B, H)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                batch_first=True)

        return outputs, (state_h, state_c)


class Attention(nn.Module):
    def __init__(self, d_q, d_v, drop_ratio=0.0):
        super(Attention, self).__init__()
        self.attn = nn.Linear(d_q+d_v, d_v)
        self.v = nn.Parameter(torch.rand(d_v))
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, Q, K, V, attn_mask):
        # Q: (B, 1, H)
        # V: (B, L, num_directions * H)
        # attn_mask: (B, L), True means mask
        k_len = K.size(1)
        q_state = Q.repeat(1, k_len, 1) # (B, L, d_q)

        attn_energies = self.score(q_state, K) # (B, L)

        attn_energies.masked_fill_(attn_mask, -1e12)

        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
        attn_weights = self.dropout(attn_weights)

        # (B, 1, L) * (B, L, d_v)  -> (B, 1, d_v)
        context = attn_weights.bmm(V)

        return context, attn_weights


    def score(self, query, memory):
        # query (B, L, d_q)
        # memory (B, L, d_k)

        # (B, L, d_q+d_v) -> (B, L, d_v)
        energy = torch.tanh(self.attn(torch.cat([query, memory], 2)))
        energy = energy.transpose(1, 2)  # (B, d_v, L)

        v = self.v.repeat(memory.size(0), 1).unsqueeze(1)  # (B, 1, d_v)
        energy = torch.bmm(v, energy)  # (B, 1, d_v) * (B, d_v, L) -> (B, 1, L)
        return energy.squeeze(1)  # (B, L)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=1, drop_ratio=0.2, attn_drop_ratio=0.1):
        super(Decoder, self).__init__()

        # for bidir encoder
        self.dropout_layer = nn.Dropout(drop_ratio)
        self.attention = Attention(d_q=hidden_size, d_v=hidden_size*2,
            drop_ratio=attn_drop_ratio)

        # hidden_size for attention output, input_size for emb
        self.rnn = nn.LSTM(256, hidden_size, n_layers,
            dropout=(0 if n_layers == 1 else drop_ratio),  batch_first=True)

        self.dec_merge = nn.Linear(input_size, 256)

    def forward(self, emb_inp, last_state, enc_outs, attn_mask, feature):
        # emb_inp: (B, 1, emb_size)
        # enc_outs: (B, L, H*2)
        # feature: (B, feature_size)
        embedded = self.dropout_layer(emb_inp)

        # use h_t as query
        # last_state[0]: (1, B, H)
        query = last_state[0].transpose(0,1) # (B, 1, H)

        # context: (B, 1, H*2)
        context, attn_weights = self.attention(query, enc_outs, enc_outs, attn_mask)

        rnn_input = torch.cat([embedded, context, feature.unsqueeze(1)], 2)

        x = self.dec_merge(rnn_input)
        output, state = self.rnn(x, last_state)

        output = output.squeeze(1)  # (B, 1, N) -> (B, N)
        return output, state, attn_weights



class InverseAutoregressiveBlock(nn.Module):
    """The Inverse Autoregressive Flow block,
    https://arxiv.org/abs/1606.04934"""
    def __init__(self, n_z, n_h, n_made):
        super(InverseAutoregressiveBlock, self).__init__()

        # made: take as inputs: z_{t-1}, h; output: m_t, s_t
        self.made = MADE(num_input=n_z, num_output=n_z * 2,
                     num_hidden=n_made, num_context=n_h)
        self.sigmoid_arg_bias = nn.Parameter(torch.ones(n_z) * 2)


    def forward(self, prev_z, h):
        '''
        prev_z: z_{t-1}
        h: the context
        '''
        m, s = torch.chunk(self.made(prev_z, h), chunks=2, dim=-1)
        # the bias is used to make s sufficiently positive
        #   see Sec. 4 in (Kingma et al., 2016) for more details
        s = s + self.sigmoid_arg_bias
        sigma = torch.sigmoid(s)
        z = sigma * prev_z + (1 - sigma) * m

        log_det = -F.logsigmoid(s)

        return z, log_det



class IAF(nn.Module):
    """docstring for IAF"""
    def __init__(self, n_z, n_h, n_made, flow_depth):
        super(IAF, self).__init__()
        self._flow_depth = flow_depth
        self._flows = nn.ModuleList(
            [InverseAutoregressiveBlock(n_z, n_h, n_made)
             for _ in range(0, flow_depth)])

        self._reverse_idxes = np.array(np.arange(0, n_z)[::-1])

    def _do_reverse(self, v):
        return v[:, self._reverse_idxes]

    def forward(self, z, h):
        total_log_det = torch.zeros_like(z, device=z.device)
        for i, flow in enumerate(self._flows):
            z, log_det = flow(z, h)
            z = self._do_reverse(z)
            total_log_det += log_det
        return z, total_log_det

#---------------------------------------

class Criterion(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self._criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)
        self._pad_idx = pad_idx


    def forward(self, outputs, targets, truncate=False):
        # outputs: (B, L, V)
        # targets: (B, L)
        # truncate: sometimes outputs may be longer than targets,
        #   we truncate outputs to the length of targets
        vocab_size = outputs.size(-1)
        tgts = targets.contiguous().view(-1) # tgts: (N)

        if truncate:
            tgt_len = targets.size(1)
            outs = outputs[:, 0:tgt_len, :].contiguous().view(-1, vocab_size) # outs: (N, V)
        else:
            outs = outputs.contiguous().view(-1, vocab_size) # outs: (N, V)

        non_pad_mask = tgts.ne(self._pad_idx)

        loss = self._criterion(outs, tgts) # [N]
        loss = loss.masked_select(non_pad_mask).mean()

        return loss
