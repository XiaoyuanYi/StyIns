# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-06 16:26:31
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from layers import Encoder, Decoder, IAF
from spectralnorm import SpectralNorm as SN
import random


def get_non_pad_mask(seq, pad_idx, device):
    # seq: (B, L)
    assert seq.dim() == 2
    # (B, L)
    mask = seq.ne(pad_idx).float()
    return mask.to(device)

def get_seq_length(seq, pad_idx, device):
    mask = get_non_pad_mask(seq, pad_idx, device)
    # mask: (B, T)
    lengths = mask.sum(dim=-1)
    lengths = lengths.long()
    return lengths

def get_attn_mask(seq, pad_idx, device):
    pad_mask = get_non_pad_mask(seq, pad_idx, device)
    attn_mask = 1 - pad_mask
    attn_mask = attn_mask.bool()
    attn_mask = attn_mask.to(device)
    return attn_mask


class Discriminator(nn.Module):
    def __init__(self, hps, device):
        super(Discriminator, self).__init__()

        filter_sizes = [2, 4, 6, 8]
        num_filters =[64, 64, 64, 64]

        self.emb_size = hps.emb_size
        self.feature_size = sum(num_filters)

        self.convs = nn.ModuleList([
            SN(nn.Conv2d(1, n, (f, self.emb_size))) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.feature2out = nn.Sequential(
            SN(nn.Linear(self.feature_size, 64)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(64, 2+1))) # M+1 classes

        self.dropout = nn.Dropout(hps.drop_ratio)
        self.activ = nn.LeakyReLU(0.2)

    def forward(self, inps):
        # inps: (B, L, emb_size)
        feature = self.get_feature(inps)
        logits = self.feature2out(self.dropout(feature))
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def get_feature(self, inps):
        embs = inps.unsqueeze(1) # (B, 1, L, emb_size)

        # features: (B, len(filter_sizes), length)
        #   each feature (B, filter_num(64), length)
        features = [self.activ(conv(embs)).squeeze(3) for conv in self.convs]
        # pools: (B, filter_size)
        pools = [F.max_pool1d(feature, feature.size(2)).squeeze(2) for feature in features]
        h = torch.cat(pools, 1)  # (B, feature_size)
        return h


class Seq2Seq(nn.Module):
    def __init__(self, hps, device):
        super(Seq2Seq, self).__init__()
        self.hps = hps
        self.device = device

        self.emb_size = hps.emb_size
        self.hidden_size = hps.hidden_size
        self.flow_h_size = hps.flow_h_size
        self.flow_depth = hps.flow_depth
        self.vocab_size = hps.vocab_size
        self.max_len = hps.max_len

        self._infor_nats = hps.infor_nats
        self._infor_groups = hps.infor_groups

        self.pad_idx = hps.pad_idx
        self.bos_idx = hps.bos_idx

        self.bos_tensor = torch.tensor(self.bos_idx, dtype=torch.long, device=device).view(1, 1)

        # we directly set the latent size to be same as that of sentence representation v_k
        self.latent_size = self.hidden_size * 2

        # componets
        self.layers = nn.ModuleDict()
        self.layers['word_embed'] = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.pad_idx)

        self.layers['source_encoder'] = Encoder(self.emb_size, self.hidden_size, drop_ratio=hps.drop_ratio)
        self.layers['style_encoder'] = Encoder(self.emb_size, self.hidden_size, drop_ratio=hps.drop_ratio)

        # decoder inpus: word embedding, encoder states, and style representation z
        self.layers['decoder'] = Decoder(self.emb_size+self.hidden_size*2+self.latent_size,
            self.hidden_size, drop_ratio=hps.drop_ratio, attn_drop_ratio=hps.attn_drop_ratio)

        self.layers['out_proj'] = nn.Linear(hps.hidden_size, hps.vocab_size)

        # MLP for calculate dec init state
        self.layers['dec_init_h'] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Tanh())
        self.layers['dec_init_c'] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Tanh())

        self.layers['flow_h_proj'] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.flow_h_size), nn.Tanh())

        # the Inverse Autoregressive Flow (IAF) module
        self.layers['iaf'] = IAF(n_z=self.latent_size, n_h=self.flow_h_size,
            n_made=hps.made_size, flow_depth=self.flow_depth)

        # --------------
        self._log2pi = torch.log(torch.tensor(2*np.pi,
            dtype=torch.float, device=device)) # log(2pi)


    # components
    def run_encoder(self, inps, encoder):
        lengths = get_seq_length(inps, self.pad_idx, self.device)
        emb_inps = self.layers['word_embed'](inps) # (B, L, emb_size)
        # enc_outs: (B, L, 2*H)
        enc_outs, enc_state = encoder(emb_inps, lengths)

        return enc_outs, enc_state


    def build_initial_gaussian(self, instances):
        # instance: (K, B, T)
        n_ins = instances.size(0)
        bsz = instances.size(1)
        length = instances.size(-1)
        flat_instances = instances.view(-1, length) # (K*B, T)

        # enc_outs: (K*B, L, 2*H)
        _, enc_state = self.run_encoder(flat_instances, self.layers['style_encoder'])

        enc_state_h = enc_state[0] # (2, K*B, H)
        points = torch.cat([enc_state_h[0, :, :], enc_state_h[1, :, :]], dim=1) # (K*B, 2*H)
        points = points.view(-1, bsz, self.hidden_size*2) # (K, B, 2*H)

        mu = points.mean(dim=0) # (B, latent_size)

        h = self.layers['flow_h_proj'](mu) # (B, flow_h_size)

        k_mu = mu.unsqueeze(0).repeat(n_ins, 1, 1) # (K, B, latent_size)
        std_sq = (points - k_mu).pow(2)
        std = torch.sqrt(std_sq.sum(dim=0) / (n_ins-1)) # unbiased estimator for variance

        return mu, std, h


    def extract_style_features(self, instances):
        mu, std, h = self.build_initial_gaussian(instances)
        eps = torch.randn_like(std)
        z0 = mu + eps * std # (B, 2*H)

        # log q(z_t|c) = log q(z_0|c) - sum log det|d_zt/dz_(t-1)|
        # log q(z) = -0.5*log(2*pi) - log(sigma) - 0.5 * eps**2
        log_qz0 = - 0.5*self._log2pi - torch.log(std) - 0.5 * eps**2

        # build flow
        z, log_det = self.layers['iaf'](z0, h)

        log_qz = log_qz0 + log_det

        return z, log_qz


    def dec_step(self, inp, state, enc_outs, attn_mask, feature):
        emb_inp = self.layers['word_embed'](inp)
        # cell_out: (B, H)
        cell_out, state, attn_weights = self.layers['decoder'](
            emb_inp, state, enc_outs, attn_mask, feature)
        out = self.layers['out_proj'](cell_out)

        normed_out = F.softmax(out, dim=1) # (B, V)

        return out, normed_out, state


    def get_emb_outs(self, out):
        # (B,V) * (V, emb_sim) -> (B, emb_dim)
        probs = F.softmax(out, dim=1)
        embs = torch.matmul(probs, self.layers['word_embed'].weight)
        return embs


    def run_decoder(self, enc_outs, dec_init_state, attn_mask, feature, bsz,
        teacher_forcing_ratio=0.0, tgt_inps=None,
        with_normed_outs=False, with_outs=False, with_emb_outs=False):

        max_len = self.max_len

        if tgt_inps is not None:
            tgt_len = tgt_inps.size(1)

        outs, normed_outs, emb_outs = None, None, None

        if with_outs:
            outs = torch.zeros(bsz, max_len, self.vocab_size, device=self.device)
        if with_normed_outs:
            normed_outs = torch.zeros(bsz, max_len, self.vocab_size, device=self.device)
        if with_emb_outs:
            emb_outs = torch.zeros(bsz, max_len, self.emb_size, device=self.device)

        state = dec_init_state
        inp = self.bos_tensor.expand(bsz, 1)
        for t in range(0, max_len):
            out, normed_out, state = self.dec_step(inp, state, enc_outs,
                attn_mask, feature)

            if with_outs:
                outs[:, t, :] = out
            if with_normed_outs:
                normed_outs[:, t, :] = normed_out # (B, V)
            if with_emb_outs:
                emb_outs[:, t, :] = self.get_emb_outs(out)


            is_teacher = random.random() < teacher_forcing_ratio

            if (tgt_inps is not None) and (t < tgt_len) and is_teacher:
                inp = tgt_inps[:, t].unsqueeze(1)
            else:
                top1 = normed_out.data.max(1)[1]
                inp = top1.unsqueeze(1)

        return outs, normed_outs, emb_outs


    def forward(self, x, x_ins, y_ins, y, teacher_forcing=1.0):
        # x: (B, T)
        # x_ins: (K, B, T)
        # y_ins: (K, B, T)
        bsz = x.size(0)


        attn_mask = get_attn_mask(x, self.pad_idx, self.device)
        enc_outs, enc_state = self.run_encoder(x, self.layers['source_encoder'])
        dec_init_state = self.calcu_dec_init(enc_state)


        # outs, normed_outs, emb_outs
        if x_ins is not None:
            x_feature, _ = self.extract_style_features(x_ins)
            x_outs, _, _ = self.run_decoder(enc_outs, dec_init_state, attn_mask,
                x_feature, bsz, with_outs=True)
        else:
            x_outs = None


        z_prior, log_p_z = self.extract_style_features(y_ins)
        y_outs_prior, _, y_emb_outs = self.run_decoder(enc_outs, dec_init_state, attn_mask,
            z_prior, bsz, tgt_inps=y[:, 1:] if y is not None else None,
            teacher_forcing_ratio=teacher_forcing/2,
            with_outs=True, with_emb_outs=True)


        # (K+1, B, T)
        if y is not None:
            # style feature (B, 2*H)
            z_post, log_p_zy = self.extract_style_features(torch.cat([y_ins, y.unsqueeze(0)], dim=0))
            y_outs_post, _, _ = self.run_decoder(enc_outs, dec_init_state, attn_mask,
            z_post, bsz, tgt_inps=y[:, 1:],
            teacher_forcing_ratio=teacher_forcing,
            with_outs=True, with_emb_outs=False)

            # kl[q(z|y,phi)||p(z|phi)] = E_q(z|y,phi)[ log q(z|y,phi) - log p(z|phi) ]
            # approximately estimated KL
            # TODO: get more samples to better estimate KL
            kl = (log_p_zy - log_p_z) # (B, latent_size)

            # objective with free bits, see IAF (Kingma et al., 2016) for more details
            kl_vae = kl.view(kl.size(0), self._infor_groups, -1)
            kl_vae = kl_vae.sum(-1).mean(0, True) # (1, 128)
            kl_obj = kl_vae.clamp(min=self._infor_nats).expand(bsz, -1)
            kl_obj = kl_obj.sum(dim=-1) # (B)


        if y is not None:
            return x_outs, y_emb_outs, y_outs_prior, y_outs_post, kl_obj
        else:
            return x_outs, y_emb_outs, y_outs_prior, None, None


    def generate_style_only(self, x, y_ins,
        teacher_forcing=0.0, y=None,
        with_emb_outs=True, with_outs=False):
        bsz = x.size(0)

        attn_mask = get_attn_mask(x, self.pad_idx, self.device)
        enc_outs, enc_state = self.run_encoder(x, self.layers['source_encoder'])
        dec_init_state = self.calcu_dec_init(enc_state)

        # feature (B, 2*H)
        y_feature, _ = self.extract_style_features(y_ins)

        y_outs, _, y_emb_outs = self.run_decoder(enc_outs, dec_init_state, attn_mask,
            y_feature, bsz,
            teacher_forcing_ratio=teacher_forcing,
            tgt_inps=y[:, 1:] if y is not None else None,
            with_outs=with_outs, with_emb_outs=with_emb_outs)

        return y_outs, y_emb_outs


    def calcu_dec_init(self, enc_state):
        # state_h: (2, B, H),  state_c: (2, B, H)
        bsz = enc_state[0].size(1)

        enc_state_h = enc_state[0].transpose(0, 1).contiguous().view(bsz, -1)
        enc_state_c = enc_state[1].transpose(0, 1).contiguous().view(bsz, -1)

        init_state_h = self.layers['dec_init_h'](enc_state_h).unsqueeze(0)
        init_state_c = self.layers['dec_init_c'](enc_state_c).unsqueeze(0)

        init_state = (init_state_h, init_state_c)

        return init_state
    #------------------------------------------

    def inference_init_encoder(self, src, ins):

        attn_mask = get_attn_mask(src, self.pad_idx, self.device)
        enc_outs, enc_state = self.run_encoder(src, self.layers['source_encoder'])

        style_feature, _ = self.extract_style_features(ins)
        dec_init_state = self.calcu_dec_init(enc_state)

        return enc_outs, dec_init_state, style_feature, attn_mask
