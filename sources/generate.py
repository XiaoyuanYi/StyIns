# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-08 00:12:07
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

from generator import Generator
from config import yelp_hps, gyafc_hps, device

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Parametrs of the generator.")
    parser.add_argument("-t", "--task", type=str, choices=['yelp', 'gyafc'], default='yelp',
        help='select the generation task.')
    parser.add_argument("-e", "--epoch", type=int, default=0,
        help="select the checkpoint to be used for generation:\n 0: the lastest ckpt, -1: all ckpts")
    return parser.parse_args()


def generate_file(generator, infile ,outfile, required_style):

    with open(infile, 'r') as fin:
        src_lines = fin.readlines()


    with open(outfile, 'w') as fout:

        for i, line in enumerate(src_lines):
            src_sent = line.strip()

            out_sent, info = generator.generate_one(src_sent, required_style)
            if len(out_sent) == 0:
                ans = info
            else:
                ans = out_sent

            fout.write(ans+"\n")
            if i % 200 == 0:
                print ("%d/%d" % (i, len(src_lines)))
                fout.flush()


def generate_file_all(generator, infile, outfile_prefix, style_id):
    epoch = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    epoch.reverse()

    for e in epoch:
        generator.reload_checkpoint(e)
        outfile = outfile_prefix + "_" + str(e) + ".txt"
        generate_file(generator, infile, outfile, style_id)


def main():
    args = parse_args()
    if args.task == 'yelp':
        hps = yelp_hps
        tgt1_prefix = "../outs/yelp_to1"
        tgt0_prefix = "../outs/yelp_to0"

        tgt1_file = "../outs/yelp_to1.txt"
        tgt0_file = "../outs/yelp_to0.txt"

        src1_file = "../inps/sentiment.test.0"
        src0_file = "../inps/sentiment.test.1"
    else:
        hps = gyafc_hps
        tgt1_prefix = "../outs/gyafc_to1"
        tgt0_prefix = "../outs/gyafc_to0"

        tgt1_file = "../outs/gyafc_to1.txt"
        tgt0_file = "../outs/gyafc_to0.txt"

        src1_file = "../inps/informal.txt"
        src0_file = "../inps/formal.txt"

    generator = Generator(hps, device)

    if args.epoch == -1:
        generate_file_all(generator, src1_file, tgt1_prefix, 1)
        generate_file_all(generator, src0_file, tgt0_prefix, 0)

    elif args.epoch >= 0:
        if args.epoch > 0:
            generator.reload_checkpoint(args.epoch)

        generate_file(generator, src1_file, tgt1_file, 1)
        generate_file(generator, src0_file, tgt0_file, 0)



if __name__ == "__main__":
    main()
