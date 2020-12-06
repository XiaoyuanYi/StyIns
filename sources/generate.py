# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-06 10:10:37
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
    epoch = [14]#[6, 8, 10, 12, 14, 16, 18]
    epoch.reverse()

    for e in epoch:
        generator.reload_checkpoint(e)
        outfile = outfile_prefix + "_" + str(e) + ".txt"
        generate_file(generator, infile, outfile, style_id)


def main():
    args = parse_args()
    if args.task == 'yelp':
        hps = yelp_hps
    else:
        hps = gyafc_hps

    generator = Generator(hps, device)

    if args.e == -1:
        generate_file_all(generator, "../inps/sentiment.test.0", "../outs/gyafc_to1", 1)
        generate_file_all(generator, "../inps/sentiment.test.1", "../outs/gyafc_to0", 0)

    elif args.e >= 0:
        if args.e > 0:
            generator.reload_checkpoint(args.e)

        generate_file(generator, "../inps/sentiment.test.0", "../outs/gyafc_to1.txt", 1)
        generate_file(generator, "../inps/sentiment.test.1", "../outs/gyafc_to0.txt", 0)



if __name__ == "__main__":
    main()
