# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-18 20:21:05
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

import torch
import torch.nn.functional as F

import utils

from layers import Criterion
from scheduler import ISRScheduler
from logger import PreLogger


class DAETrainer(object):

    def __init__(self, hps, device):
        self.hps = hps
        self.device = device


    def run_validation(self, epoch, generator, criterion, tool, lr):
        print("run validation...")

        logger = PreLogger('valid')
        logger.set_batch_num(tool.valid_batch_num)
        logger.set_log_path(self.hps.pre_valid_log_path)
        logger.set_rate('learning_rate', lr)

        for step in range(0, tool.valid_batch_num):

            # (x, x_ins, y_ins, x_id, y_id, y)
            batch = tool.get_valid_batch(step)
            x = batch[0].to(self.device)
            x_ins = batch[1].to(self.device)
            y_ins = batch[2].to(self.device)
            x_id, y_id = batch[3], batch[4]
            x_tgt = batch[6].to(self.device)


            # -------------------------------------------------------------
            recon_loss, _ = self.step_gen(x, x_ins, x_tgt, generator,
                None, criterion, tool, valid=True)

            logger.add_recon_loss(recon_loss)

        logger.print_log(epoch=epoch)


    # ------------------------------------
    def step_gen(self, x, x_ins, x_tgt, generator,
        optimizerGen, criterion, tool, valid=False):

        if not valid:
            optimizerGen.zero_grad()

        # y_emb_outs: (B, T, emb_size)
        x_outs, _ = generator.generate_style_only(x, x_ins,
            with_emb_outs=False, with_outs=True)


        # reconstruct loss
        recon_loss = criterion(x_outs, x_tgt[:, 1:], True)

        #-------------------------------------------
        loss = recon_loss
        loss = loss.mean()

        if (not valid) and (not torch.isnan(loss).item()):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), self.hps.clip_grad_norm)
            optimizerGen.step()

        return recon_loss.mean().item(), x_outs


    # ----------------------------------------------------------------
    def run_train(self, generator, tool, optimizerGen, criterion, logger):

        logger.set_start_time()

        for step in range(0, tool.train_batch_num):

            # (x, x_ins, y_ins, x_id, y_id, y)
            batch = tool.get_train_batch(step)
            x = batch[0].to(self.device)
            x_ins = batch[1].to(self.device)
            y_ins = batch[2].to(self.device)
            x_id, y_id = batch[3], batch[4]
            x_tgt = batch[6].to(self.device)

            # -------------------------------------------------------------
            # train generator
            recon_loss, x_outs = self.step_gen(x, x_ins, x_tgt, generator,
                optimizerGen, criterion, tool)

            logger.add_recon_loss(recon_loss)

            logger.set_rate("learning_rate", optimizerGen.rate())

            # -------------------------------------------------------------
            if step % self.hps.pre_log_steps == 0:
                logger.set_end_time()
                utils.sample_dae(x, x_outs, x_tgt, x_id, self.hps.sample_num, tool)
                logger.print_log()
                logger.set_start_time()


    # -----------------------------------------------------------------
    def train(self, generator, tool):
        print ("using device: %s" %(self.device))

        # build training and validation data
        tool.open_corruption()
        tool.set_batch_size(self.hps.pre_batch_size)
        tool.build_train_data(self.hps.unpaired_train_data, None, None)
        tool.build_valid_data(self.hps.unpaired_valid_data, None, None)

        print ("train batch num: %d" % (tool.train_batch_num))
        print ("valid batch num: %d" % (tool.valid_batch_num))

        # training logger
        logger = PreLogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.pre_log_steps)
        logger.set_log_path(self.hps.pre_train_log_path)
        logger.set_rate('learning_rate', 0.0)

        # build optimizer
        optGen = torch.optim.AdamW(generator.parameters(),
            lr=0.0, betas=(0.9, 0.999), weight_decay=self.hps.weight_decay)

        optimizerGen = ISRScheduler(optimizer=optGen, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr, init_lr=self.hps.init_lr, beta=0.75)

        # ----------------------------------------------
        criterion = Criterion(self.hps.pad_idx)

        generator.train()

        tool.set_batch_size(self.hps.pre_batch_size)

        print ("In training mode: %d" % (generator.training))
        #input("Please check the parameters and press enter to continue>")

        # --------------------------------------------------------------------------------------
        for epoch in range(1, self.hps.pre_epoches+1):

            self.run_train(generator, tool, optimizerGen, criterion, logger)

            if epoch % self.hps.pre_valid_epoches == 0:
                print("run validation...")
                generator.eval()
                print ("in training mode: %d" % (generator.training))
                self.run_validation(epoch, generator, criterion, tool, optimizerGen.rate())
                generator.train()
                print ("validation Done, mode: %d" % (generator.training))

            if (self.hps.pre_save_epoches >= 1) and (epoch % self.hps.pre_save_epoches) == 0:
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint_generator(self.hps.ckpt_path,
                    generator, optimizerGen, epoch, prefix="dae_")

            logger.add_epoch()

            print("shuffle data...")
            tool.shuffle_training_data()