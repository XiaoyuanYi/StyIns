# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-20 09:16:41
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''
from styins_trainer import StyInsTrainer
from lm_trainer import LMTrainer
from dae_trainer import DAETrainer

from graphs import Seq2Seq, Discriminator
from tool import Tool
from config import device, yelp_hps, gyafc_hps
import utils

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="The parametrs of StyIns.")
    parser.add_argument("-t", "--task", type=str, choices=['yelp', 'gyafc'], default='yelp',
        help='select the generation task.')
    return parser.parse_args()



def pretrain(generator, tool, hps, specified_device, pretrain_method):
    if pretrain_method == 'lm':
        pre_trainer = LMTrainer(hps, specified_device)
        print ("pretraining...")
        pre_trainer.train(tool)
        print ("pretraining done!")
    else:
        pre_trainer = DAETrainer(hps, specified_device)
        print ("pretraining...")
        pre_trainer.train(generator, tool)
        print ("pretraining done!")



def train(generator, discriminator, tool, hps, specified_device, pretrain_method):
    '''
    last_epoch = utils.restore_checkpoint_generator(hps.ckpt_path, specified_device,
        generator, None, None)

    if last_epoch is not None:
         print ("checkpoints exsit! directly recover!")
    else:
         print ("checkpoints not exsit! train from scratch!")
    '''
    # for lm pretraing, we initialize parameters of the encoders and decoder
    #   with the those of the lm
    if pretrain_method == 'lm':
        utils.restore_checkpoint_overlap(hps.ckpt_path, generator, specified_device)

    # for dae pretraing, we directly reuse the pretrained generator
    styins_trainer = StyInsTrainer(hps, specified_device)
    styins_trainer.train(generator, discriminator, tool)


def main():
    args = parse_args()

    if args.task == 'yelp':
        hps = yelp_hps
    else:
        hps = gyafc_hps

    tool = Tool(vocab_file=hps.vocab_path, n_ins=hps.n_ins,
        batch_size=hps.batch_size, max_len=hps.max_len, r_superv=hps.r_superv,
        corrupt_ratio=hps.corrupt_ratio)

    tool.build_vocab([hps.unpaired_train_data, hps.paired_train_data])

    vocab_size = tool.vocabulary_size
    pad_idx = tool.pad_idx
    bos_idx = tool.bos_idx
    assert vocab_size > 0 and pad_idx >=0 and bos_idx >= 0
    hps = hps._replace(vocab_size=vocab_size, pad_idx=pad_idx, bos_idx=bos_idx)

    print ("hyper-patameters:")
    print (hps)
    input ("please check the hyper-parameters, and then press any key to continue >")

    generator = Seq2Seq(hps, device)
    generator = generator.to(device)

    discriminator = Discriminator(hps, device)
    discriminator = discriminator.to(device)

    pretrain_method = hps.pretrain_method

    pretrain(generator, tool, hps, device, pretrain_method)
    train(generator, discriminator, tool, hps, device, pretrain_method)


if __name__ == "__main__":
    main()
