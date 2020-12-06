# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-06 09:56:16
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

import os
import torch
import random


def save_checkpoint_generator(model_dir, gen_model, gen_optim, epoch, prefix=""):
    # save model state dict
    model_state_path = os.path.join(model_dir, prefix + "model_ckpt_{}e.tar".format(epoch))

    saved_dic = {
        'epoch': epoch,
        'gen_model_state_dict': gen_model.state_dict()
    }

    if gen_optim is not None:
        saved_dic['gen_optimizer_state_dict'] = gen_optim.state_dict()

    torch.save(saved_dic, model_state_path)

    # write checkpoint information
    log_path = os.path.join(model_dir, "ckpt_list.txt")
    fout = open(log_path, 'a')
    fout.write(prefix + "model_ckpt_{}e.tar".format(epoch)+"\n")
    fout.close()


def restore_checkpoint_generator(model_dir, device, model, optimizer=None,
    specified_epoch=None, prefix=""):

    if specified_epoch is not None:
       restore_ckpt = prefix + "model_ckpt_{}e.tar".format(specified_epoch)

    else:
        ckpt_list_path = os.path.join(model_dir, "ckpt_list.txt")
        if not os.path.exists(ckpt_list_path):
            print ("checkpoint list not exists, creat new one!")
            return None

        # get latest ckpt name
        with open(ckpt_list_path, 'r') as fin:
            restore_ckpt = fin.readlines()[-1].strip()


    restore_ckpt_path = os.path.join(model_dir, restore_ckpt)
    if not os.path.exists(restore_ckpt_path):
        print ("latest checkpoint not exists!")
        return None

    print ("restore checkpoint from %s" % (restore_ckpt_path))
    print ("loading...")
    checkpoint = torch.load(restore_ckpt_path, map_location=device)

    print ("load state dic...")
    model.load_state_dict(checkpoint['gen_model_state_dict'])
    epoch = checkpoint['epoch']

    if optimizer is not None:
        print ("load optimizer dic...")
        optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    print ("ok!")
    return epoch

#------------------------------------------------------
def restore_checkpoint_multiple(model_dir, device,
    gen_model, dis_model, gen_optim, dis_optim):
    ckpt_list_path = os.path.join(model_dir, "ckpt_list.txt")
    if not os.path.exists(ckpt_list_path):
        print ("checkpoint list not exists, creat new one!")
        return None

    # get latest ckpt name
    fin = open(ckpt_list_path, 'r')
    latest_ckpt_path = fin.readlines()[-1].strip()
    fin.close()

    latest_ckpt_path = os.path.join(model_dir, latest_ckpt_path)
    if not os.path.exists(latest_ckpt_path):
        print ("latest checkpoint not exists!")
        return None

    print ("restore checkpoint from %s" % (latest_ckpt_path))
    print ("loading...")
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    #checkpoint = torch.load(latest_ckpt_path)
    print ("load state dic...")
    gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
    dis_model.load_state_dict(checkpoint['dis_model_state_dict'])

    if gen_optim is not None:
        print ("load optimizer dic...")
        gen_optim.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        dis_optim.load_state_dict(checkpoint['dis_optimizer_state_dict'])

    epoch = checkpoint['epoch']

    return epoch


def save_checkpoint_multiple(model_dir, gen_model, dis_model,
    gen_optim, dis_optim, epoch):
    # save model state dict
    model_state_path = os.path.join(model_dir, "model_ckpt_{}e.tar".format(epoch))

    saved_dic = {
        'epoch': epoch,
        'gen_model_state_dict': gen_model.state_dict(),
        'dis_model_state_dict': dis_model.state_dict(),

        'gen_optimizer_state_dict': gen_optim.state_dict(),
        'dis_optimizer_state_dict': dis_optim.state_dict()
    }

    torch.save(saved_dic, model_state_path)

    # write checkpoint information
    log_path = os.path.join(model_dir, "ckpt_list.txt")
    fout = open(log_path, 'a')
    fout.write("model_ckpt_{}e.tar".format(epoch)+"\n")
    fout.close()


#----------------------------------------------------------------
def sample(x, outs_prior, outs_post, y, x_id, y_id, sample_num, tool):
    # x: (B, L)
    # logits: (B, L, V)
    bsz = x.size(0)

    prior_probs = torch.nn.functional.softmax(outs_prior, dim=-1)
    if outs_post is not None:
        post_probs = torch.nn.functional.softmax(outs_post, dim=-1)

    sample_num = min(sample_num, bsz)

    # select some random examples
    idx_vec = random.sample(list(range(0, bsz)), sample_num)

    for idx in idx_vec:
        # build lines
        x_len = x.size(1)
        x_indices = [x[idx, t].item() for t in range(0, x_len)]
        x_line = tool.indices2sent(x_indices, True, True)

        prior_len = prior_probs.size(1)
        prior_line = [prior_probs[idx, t, :].cpu().data.numpy() for t in range(0, prior_len)]
        prior_line = tool.greedy_search(prior_line)

        if y is not None:
            y_len = y.size(1)
            y_indices = [y[idx, t].cpu().item() for t in range(0, y_len)]
            y_line = tool.indices2sent(y_indices, True, True)

            post_len = post_probs.size(1)
            post_line = [post_probs[idx, t, :].cpu().data.numpy() for t in range(0, post_len)]
            post_line = tool.greedy_search(post_line)

        print (str(x_id) + "-->" + str(y_id))
        print("x: " + x_line + "\n")
        print("prior out: " + prior_line + "\n")
        if y is not None:
            print("post out: " + post_line + "\n")
            print("y: " + y_line + "\n")
        print ("")




def sample_pre(x, x_outs, x_id, sample_num, tool):
    # x: (B, L)
    # logits: (B, L, V)
    bsz = x.size(0)

    x_out_probs = torch.nn.functional.softmax(x_outs, dim=-1)

    sample_num = min(sample_num, bsz)

    # select some random examples
    idx_vec = random.sample(list(range(0, bsz)), sample_num)


    for idx in idx_vec:
        # build lines
        x_len = x.size(1)
        x_indices = [x[idx, t].item() for t in range(0, x_len)]
        x_line = tool.indices2sent(x_indices, True, False)


        x_out_len = x_out_probs.size(1)
        x_out_line = [x_out_probs[idx, t, :].cpu().data.numpy() for t in range(0, x_out_len)]
        x_out_line = tool.greedy_search(x_out_line)


        print ("style id: " + str(x_id))
        print("x: " + x_line + "\n")
        print("x out: " + x_out_line + "\n")
        print ("")


def print_parameter_list(model):
    params = list(model.named_parameters())
    print ("params num: %d" % (len(params)))
    for name, param in params:
        print(name, param.size())