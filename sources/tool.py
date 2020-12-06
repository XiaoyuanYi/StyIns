# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-05 16:56:02
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

import numpy as np
import random
import copy
import json
import torch
import os
import gc

from collections import Counter, OrderedDict


class Tool(object):
    def __init__(self, vocab_file, n_ins, batch_size, max_len, r_superv=0.0,
        min_freq=0, max_vocab_size=None, corrupt_ratio=0.0):
        '''
        n_ins: the number of instances used to construct the style space
        r_superv: the ratior of paired sentences to be used
        '''
        self._n_ins = n_ins
        self._bsz = batch_size
        self._max_len = max_len
        self._r_superv = r_superv

        self._pad = '<pad>'
        self._unk = '<unk>'
        self._eos = '<eos>'
        self._bos = '<bos>'
        self._num = '__NUM'

        self._special_tokens = \
            [self._pad, self._unk, self._eos, self._bos, self._num]

        #-----------------------------------
        self._min_freq = min_freq
        self._max_vocab_size = max_vocab_size
        self._vocab_file = vocab_file

        self._corrupt_ratio = corrupt_ratio
        self._corruption = False

    # ---------------------------------
    @property
    def pad_idx(self):
        return self._tok2idxDic[self._pad]

    @property
    def bos_idx(self):
        return self._tok2idxDic[self._bos]

    @property
    def eos_idx(self):
        return self._tok2idxDic[self._eos]

    @property
    def unk_idx(self):
        return self._tok2idxDic[self._unk]

    @property
    def num_idx(self):
        return self._tok2idxDic[self._num]


    @property
    def vocabulary_size(self):
        return len(self._tok2idxDic)

    def get_vocab(self):
        return self._tok2idxDic

    def get_ivocab(self):
        return self._idx2tokDic

    def open_corruption(self):
        self._corruption = True

    def close_corruption(self):
        self._corruption = False

    def set_batch_size(self, batch_size):
        assert 0 < batch_size < 1024
        self._bsz = batch_size

    # -----------------------------------
    # Tool functions
    def sent2tokens(self, sent):
        tokens = sent.strip().split(" ")
        return tokens

    def tokens2sent(self, tokens):
        return " ".join(tokens).strip()

    def idx2token(self, idx):
        return self._idx2tokDic[idx]

    def token2idx(self, token):
        if token in self._tok2idxDic:
            return self._tok2idxDic[token]
        else:
            return self.unk_idx

    def indices2tokens(self, indices):
        return [self.idx2token(idx) for idx in indices]

    def tokens2indices(self, tokens):
        return [self.token2idx(token) for token in tokens]


    def batch2tensor(self, sents):
        batch_size = len(sents)
        sent_len = max([len(sent) for sent in sents])
        tensor = torch.zeros(batch_size, sent_len, dtype=torch.long)
        for i, sent in enumerate(sents):
            assert len(sent) == sent_len
            for j, idx in enumerate(sent):
                tensor[i][j] = idx
        return tensor


    def sent2tensor(self, sent):
        indices = self.sent2indices(sent)
        return self.batch2tensor([indices])


    def indices2sent(self, indices, truncate=True, exclude=False):
        tokens = self.indices2tokens(indices)
        if truncate and self._eos in tokens:
            tokens = tokens[:tokens.index(self._eos)]

        if exclude:
            tokens = [token for token in tokens if token not in self._special_tokens]

        return self.tokens2sent(tokens)


    def sent2indices(self, sent):
        tokens = self.sent2tokens(sent)
        return self.tokens2indices(tokens)

    # --------------------------------------------------
    def greedy_search(self, probs, as_indices=False):
        # probs: (B, L, V)
        out_indices = [int(np.argmax(prob, axis=-1)) for prob in probs]
        sent = self.indices2sent(out_indices, True, True)
        if as_indices:
            return self.sent2indices(sent)
        else:
            return sent

    # ----------------------------------------------------------------
    def add_token(self, token):
        if token not in self._tok2idxDic:
            self._idx2tokDic.append(token)
            self._tok2idxDic[token] = len(self._idx2tokDic) - 1


    def build_vocab(self, source_files):
        self._idx2tokDic = []
        self._tok2idxDic = OrderedDict()

        # add special tokens
        for token in self._special_tokens:
            self.add_token(token)

        if os.path.exists(self._vocab_file):
            print('loading existing vocab from {}'.format(self._vocab_file))
            with open(self._vocab_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    self.add_token(line.strip())
            print('final vocab size {}'.format(len(self._tok2idxDic)))
            return

        # build vocabulary from source text files
        print('building vocab from the corpus, min_freq={}, max_size={}'.format(
            self._min_freq, self._max_vocab_size))

        counter = Counter()
        self.count_file(source_files, counter=counter)

        # NOTE: words must be ordered in term of frequency
        for token, cnt in counter.most_common(self._max_vocab_size):
            if cnt < self._min_freq: break
            self.add_token(token)

        print('final vocab size {} from {} unique tokens'.format(
            len(self._tok2idxDic), len(counter)))

        # save vocabulary
        print ('save vocabulary into {}'.format(self._vocab_file))
        with open(self._vocab_file, 'w', encoding='utf-8') as fout:
            for token in self._tok2idxDic.keys():
                fout.write(token+"\n")


    def count_file(self, paths, counter):

        for path in paths:
            if path is None: continue
            assert os.path.exists(path)
            print ("building vocabulary from {}".format(path))
            with open(path, 'r', encoding='utf-8') as fin:
                for idx, line in enumerate(fin):
                    vec = json.loads(line.strip())
                    for dic in vec:
                        tokens = self.sent2tokens(dic['sent'].strip())
                        counter.update(tokens)


    # -----------------------------------------------------------------
    def get_train_batch(self, idx):
        return self._train_data[idx]

    def get_valid_batch(self, idx):
        return self._valid_data[idx]

    def build_train_data(self, unpaired_data_path, paired_data_path, data_limit=None, combine=False):
        self._train_style_data, self._train_paired_data = \
            self.build_gen_data(unpaired_data_path, paired_data_path, self._r_superv, data_limit, combine)
        self._train_data = self.build_blocks(self._train_style_data, self._train_paired_data,
            self._corruption)
        self.train_batch_num = len(self._train_data)


    def build_valid_data(self, unpaired_data_path, paired_data_path, data_limit=None, combine=False):
        self._valid_style_data, self._valid_paired_data = \
            self.build_gen_data(unpaired_data_path, paired_data_path, 1.0, data_limit, combine)
        self._valid_data = self.build_blocks(self._valid_style_data, self._valid_paired_data,
            False)
        self.valid_batch_num = len(self._valid_data)


    def load_data(self, path):
        if path is None:
            return None

        print ("reading %s ..." % (path))
        data = []

        skip_count = 0
        with open(path, 'r', encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                vec = json.loads(line.strip())
                new_vec = []
                skip_flag = False
                for dic in vec:
                    label = dic['label']
                    sent = self.sent2indices(dic['sent'])
                    if len(sent) > self._max_len - 1:
                        skip_flag = True
                        break
                    new_vec.append((sent, label))

                if skip_flag:
                    skip_count += 1
                    continue
                data.append(new_vec)

        print ("data num: %d, skip count: %d" % (len(data), skip_count))
        return data



    def build_gen_data(self, unpaired_data_path, paired_data_path, r_superv, data_limit, combine):
        '''
        build data as batches.
        NOTE: please run build_vocab() at first.
        '''
        unpaired_data = self.load_data(unpaired_data_path)
        ori_paired_data = self.load_data(paired_data_path)

        # TODO: support more than two styles
        style_data = [ [], [] ]

        paired_data = None
        if (ori_paired_data is not None) and (r_superv > 0):

            paired_data = []
            for vec in ori_paired_data:
                assert vec[0][1] != vec[1][1]
                pair = ["", ""]
                pair[vec[0][1]] = vec[0][0]
                pair[vec[1][1]] = vec[1][0]

                paired_data.append(pair)

                if combine:
                    for d in vec:
                        style_data[d[1]].append(d[0])


        for instance in unpaired_data:
            for pair in instance:
                style_data[pair[1]].append(pair[0])


        if data_limit is not None:
            style_data[0] = style_data[0][0:data_limit]
            style_data[1] = style_data[1][0:data_limit]
            if paired_data is not None:
                paired_data = paired_data[0:data_limit]

        if paired_data is not None:
            # use specified ratio of paired data
            paired_data = paired_data[0:int(len(paired_data) * r_superv)]
            if r_superv == 0:
                paired_data = None

        #print (len(style_data[0]), len(style_data[1]))
        #print (len(paired_data))
        return style_data, paired_data



    def extract_batch(self, data, bi):
        if bi * self._bsz > len(data):
            sents = random.sample(data, self._bsz)
            return sents

        sents = data[bi*self._bsz : (bi+1)*self._bsz]
        if len(sents) < self._bsz:
            sents = sents + random.sample(data, self._bsz-len(sents))
        return sents


    # -------------------------------------------------------------------
    def build_blocks(self, style_data, paired_data, corruption):
        data_num = max(len(style_data[0]), len(style_data[1]))
        batch_num = int(np.ceil(data_num / float(self._bsz)))

        interval = -1
        if paired_data is not None:
            # we insert paired batch with equal interval
            superv_batch_num = int(np.ceil(len(paired_data) / float(self._bsz)))
            interval = int(batch_num // (superv_batch_num))
            print ("insert interval: %d" % (interval))
            #print ( len(paired_data), superv_batch_num, batch_num)
            batch_num += superv_batch_num


        block_data = []

        # in each epoch, we randomly assign the transfer direction
        half_size = int(batch_num//2) + 1
        style_ids = [(0,1)] * half_size + [(1,0)]*half_size
        random.shuffle(style_ids)

        superv_count, unsuperv_count = 0, 0
        for bi in range(0, batch_num):
            (x_id, y_id) = style_ids[bi]
            assert x_id != y_id

            if (interval > 0) and ((bi+1) % interval == 0):

                batch = self.extract_batch(paired_data, superv_count)

                x_sents = [d[x_id] for d in batch]
                y_sents = [d[y_id] for d in batch]

                # style instances, (K*B, T)
                x_ins = self.sample_instances(style_data[x_id], x_sents)
                y_ins = self.sample_instances(style_data[y_id], y_sents)

                x_batch = self._build_batch(x_sents) # (B, T)
                # (K*B, T) -> (K, B, T)
                x_ins_batch = self._build_batch(x_ins).view(self._n_ins, self._bsz, -1)

                max_len = max([len(sent) for sent in y_ins + y_sents])
                y_ins_batch = self._build_batch(y_ins, max_len).view(self._n_ins, self._bsz, -1)

                # build target batch
                y_batch = self._build_batch(y_sents, max_len) # (B, T)
                assert y_ins_batch.size(2) == y_batch.size(1)
                superv_count += 1

            else:
                x_sents = self.extract_batch(style_data[x_id], unsuperv_count)
                x_batch = self._build_batch(x_sents, corrupt=corruption)
                y_sents = []

                # style instances, (K*B, T)
                x_ins = self.sample_instances(style_data[x_id], x_sents)
                y_ins = self.sample_instances(style_data[y_id], y_sents)

                x_ins_batch = self._build_batch(x_ins).view(self._n_ins, self._bsz, -1) # (K, B, T)
                y_ins_batch = self._build_batch(y_ins).view(self._n_ins, self._bsz, -1) # (K, B, T)
                y_batch = None
                unsuperv_count += 1

            block_data.append((x_batch, x_ins_batch, y_ins_batch, x_id, y_id, y_batch))

        return block_data


    def shuffle_training_data(self):

        random.shuffle(self._train_style_data[0])
        random.shuffle(self._train_style_data[1])

        if self._train_paired_data is not None:
            random.shuffle(self._train_paired_data)

        self._train_data = []
        gc.collect()

        self._train_data = self.build_blocks(self._train_style_data, self._train_paired_data, self._corruption)
        self.train_batch_num = len(self._train_data)


    def sample_instances(self, style_data, confict_sents):

        def build_idxes2str(idxes):
            vs = [str(v) for v in idxes]
            return " ".join(vs)

        confict_dic = {}
        for sent in confict_sents:
            sent_str = build_idxes2str(sent)
            confict_dic[sent_str] = 1

        ins_vec = []
        num = self._n_ins * self._bsz
        while len(ins_vec) < num:
            sents = random.sample(style_data, num)
            for sent in sents:
                sent_str = build_idxes2str(sent)
                if sent_str in confict_dic:
                    continue
                ins_vec.append(sent)
                confict_dic[sent_str] = 1
        ins_vec = ins_vec[0:num]
        assert len(ins_vec) == num
        return ins_vec



    # ------------------------------------------------------------
    # ------------------------------------------------------------
    def _build_batch(self, sents, max_len=None, corrupt=False):
        #print (len(instances))
        batch = self._get_batch_sen(sents, len(sents), True, max_len, corrupt)
        batch_tensor = self.batch2tensor(batch)
        return batch_tensor


    def _do_corruption(self, inp):
        # corrupt the sequence by setting some tokens as UNK
        m = int(np.ceil(len(inp) * self._corrupt_ratio))
        m = min(m, len(inp))
        m = max(1, m)

        unk_id = self.unk_idx

        corrupted_inp = copy.deepcopy(inp)
        pos = random.sample(list(range(0, len(inp))), m)
        for p in pos:
            corrupted_inp[p] = unk_id

        return corrupted_inp


    def _get_batch_sen(self, sents, batch_size,
        with_BE=False, required_max_len=None, corrupt=False):

        assert len(sents) == batch_size
        max_len = max([len(sent) for sent in sents])

        if required_max_len is not None:
            max_len = required_max_len

        batched_sents = []

        for i in range(batch_size):
            sent = sents[i]

            if corrupt:
                sent = self._do_corruption(sent)

            pad_size = max_len - len(sent)

            pads = [self.pad_idx] * pad_size

            if with_BE:
                new_sent = [self.bos_idx] + sent + [self.eos_idx] + pads
            else:
                new_sent = sent + pads

            batched_sents.append(new_sent)

        return batched_sents


    def rebuild_outs(self, logits):
        # logits (B, L, V)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        bsz, tgt_len = probs.size(0), probs.size(1)
        prob_matrix = probs.cpu().data.numpy()

        sequences = []

        for idx in range(0, bsz):
            # Build lines
            out = [prob_matrix[idx, t, :] for t in range(0, tgt_len)]
            idxes = self.greedy_search(out, as_indices=True)
            sequences.append(idxes)

        # build batch
        batch = self._get_batch_sen(sequences, len(sequences), True)

        # (B, T)
        tensor = self.batch2tensor(batch)
        return tensor


    # -------------------------------------------------------------
    # tools for generation
    def build_inference_src(self, sen, beam_size=1):
        indices = self.sent2indices(sen)

        batch = [indices for _ in range(0, beam_size)]
        src = self._build_batch(batch, None, False)

        return src


    def build_inference_instances(self, style_id, confict_sen_str):
        confict_sents = [self.sent2indices(confict_sen_str)]
        ins = self.sample_instances(self._valid_style_data[style_id], confict_sents)
        batch = self._build_batch(ins)
        tensor = self.batch2tensor(batch) # (K, L)

        return tensor.unsqueeze(1) # (K, 1, L)