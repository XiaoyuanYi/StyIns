# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-06 17:23:05
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import numpy as np
import torch
import json


''''construction for Evaluator'''
class Validator(object):

    def __init__(self, valid_file, device, num=10):
        self._device = device

        # reading validation data for testing BLEU
        with open(valid_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        lines = lines[0:num] + lines[-num:]

        self._inps0to1, self._inps1to0 = [], []
        self._refs0to1, self._refs1to0 = [], []


        for line in lines:
            vec = json.loads(line.strip())

            labels = [ dic['label'] for dic in vec ]
            sents = [ dic['sent'] for dic in vec ]

            inp = sents[0]
            refs = [sent.strip().split(" ") for sent in sents[1:]]

            x_id, y_id = labels[0], labels[1]

            if x_id == 0:
                self._inps0to1.append(inp)
                self._refs0to1.append(refs)

            elif x_id == 1:
                self._inps1to0.append(inp)
                self._refs1to0.append(refs)

        print ("validation num: %d %d" % (len(self._inps0to1), len(self._inps1to0)))


    def do_validation(self, model, tool):
        self.bos_idx = tool.bos_idx
        self.eos_idx = tool.eos_idx

        ans_0to1 = self.validation_core(model, tool, 0, 1, self._inps0to1)
        ans_1to0 = self.validation_core(model, tool, 1, 0, self._inps1to0)

        print ("generated num: %d, %d" % (len(ans_0to1), len(ans_1to0)))

        smooth = SmoothingFunction()
        bleu0to1 = corpus_bleu(self._refs0to1, ans_0to1, smoothing_function=smooth.method1)
        bleu0to1 = np.round(bleu0to1 * 100, 2)

        bleu1to0 = corpus_bleu(self._refs1to0, ans_1to0, smoothing_function=smooth.method1)
        bleu1to0 = np.round(bleu1to0 * 100, 2)

        return bleu0to1, bleu1to0


    def validation_core(self, model, tool, x_id, y_id, inps):

        ans_vec = []
        for i, inp in enumerate(inps):
            #print ("validation bleu%d" % (i))
            src = tool.build_inference_src(inp.strip(), 1)
            src = src.to(self._device) # (1, T)

            # (K, 1, L)
            ins = tool.build_inference_instances(y_id, inp.strip())
            ins = ins.to(self._device)

            with torch.no_grad():

                trans, _ = self.greedy_search(src, ins, model)

            out = tool.indices2sent(trans, True, True).strip()

            if len(out) == 0:
                out = "generation failed !"

            ans_vec.append(tool.sent2tokens(out))

        return ans_vec


    def greedy_search(self, src, ins, model):

        enc_outs, init_state, style_feature, attn_mask = model.inference_init_encoder(src, ins)

        inps = torch.tensor(self.bos_idx, dtype=torch.long, device=self._device).view(1,1)
        states = init_state
        length = src.size(1)

        trans = []
        costs = 0.0
        for k in range(0, length*2):

            _, probs, states = model.dec_step(inps, states, enc_outs, attn_mask, style_feature)
            top1 = probs.data.max(1)[1]
            inps = top1.unsqueeze(1)

            token = top1.item()
            trans.append(token)
            costs -= np.log(probs[0, token].item()+1e-12)

            if token == self.eos_idx:
                break

        return trans, costs