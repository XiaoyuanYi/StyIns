# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-06 16:26:40
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

import torch
import torch.nn.functional as F

import utils
import numpy as np
import random

from graphs import Seq2Seq
from tool import Tool


''''construction for Evaluator'''
class Generator(object):

    def __init__(self, hps, device, epoch=None):
        # construct HParams
        self.device = device

        self.tool = Tool(vocab_file=hps.vocab_path, n_ins=hps.n_ins,
            batch_size=hps.batch_size, max_len=hps.max_len, r_superv=hps.r_superv,
            corrupt_ratio=0.0)

        self.tool.build_vocab([hps.unpaired_train_data, hps.paired_train_data])

        vocab_size = self.tool.vocabulary_size
        pad_idx = self.tool.pad_idx
        bos_idx = self.tool.bos_idx
        assert vocab_size > 0 and pad_idx >=0 and bos_idx >= 0
        self.hps = hps._replace(vocab_size=vocab_size, pad_idx=pad_idx, bos_idx=bos_idx)

        # load model
        model = Seq2Seq(self.hps, device)

        # load
        utils.restore_checkpoint_generator(hps.ckpt_path, device, model, optimizer=None,
            specified_epoch=epoch, prefix="")

        model.eval()

        self.model = model.to(device)

        self.tool.close_corruption()
        # we use unpaired validation data to build stylistic instances for generation
        self.tool.build_valid_data(self.hps.unpaired_valid_data, None)


    def reload_checkpoint(self, epoch):
        utils.restore_checkpoint_generator(self.hps.ckpt_path, self.device, self.model, optimizer=None,
            specified_epoch=epoch, prefix="")

        self.model = self.model.to(self.device)
        self.model.eval()


    def greedy_search(self, src, ins):

        enc_outs, init_state, style_feature, attn_mask = self.model.inference_init_encoder(src, ins)

        inps = torch.tensor(self.tool.bos_idx, dtype=torch.long, device=self.device).view(1,1)
        states = init_state
        length = src.size(1)

        trans = []
        costs = 0.0
        for k in range(0, length*2):

            _, probs, states = self.model.dec_step(inps, states, enc_outs, attn_mask, style_feature)
            top1 = probs.data.max(1)[1]
            inps = top1.unsqueeze(1)

            token = top1.item()
            trans.append(token)
            costs -= np.log(probs[0, token].item()+1e-12)

            if token == self.tool.eos_idx:
                break

        return trans, costs


    def generate_one(self, src_sent, style_id):

        src = self.tool.build_inference_src(src_sent.strip())
        src = src.to(self.device) # (1, T)

        # (K, 1, L)
        ins = self.tool.build_inference_instances(style_id, src_sent.strip())
        ins = ins.to(self.device)


        trans, costs = self.greedy_search(src, ins)

        if len(trans) == 0:
            return "", "generation failed!"

        out_sent = self.tool.indices2sent(trans)

        return out_sent, "ok"