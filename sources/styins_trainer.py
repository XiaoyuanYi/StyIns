# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-17 17:30:16
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''

import time
import numpy as np
import random

import torch
import torch.nn.functional as F

import utils

from layers import Criterion
from validator import Validator

from decay import LinearDecay
from scheduler import ISRScheduler
from logger import StyInsLogger


class StyInsTrainer(object):

    def __init__(self, hps, device):
        self.hps = hps
        self.device = device

        # load validator
        # when paired validation data is available, we also calculate bleu
        if hps.paired_valid_data is not None:
            self.bleu_validator = Validator(hps.paired_valid_data, device)


    def run_validation(self, epoch, generator, discriminator, criterion, tool, lr):
        print("run validation...")

        logger = StyInsLogger('valid')
        logger.set_batch_num(tool.valid_batch_num)
        logger.set_log_path(self.hps.valid_log_path)
        logger.set_rate('learning_rate', lr)
        logger.set_rate('teach_ratio', self.decay_tool.get_rate())


        for step in range(0, tool.valid_batch_num):

            # (x, x_ins, y_ins, x_id, y_id, y)
            batch = tool.get_valid_batch(step)
            x = batch[0].to(self.device)
            x_ins = batch[1].to(self.device)
            y_ins = batch[2].to(self.device)
            x_id, y_id = batch[3], batch[4]

            if batch[5] is not None:
                y = batch[5].to(self.device)
            else:
                y = None

            # -------------------------------------------------------------
            recon_loss, style_loss, cycle_loss, teach_prior_loss, teach_post_loss, kl_obj,\
            _, _ = self.step_gen(x, x_ins, y_ins, y, y_id, generator,
                discriminator, None, criterion=criterion, tool=tool, valid=True)

            logger.add_recon_loss(recon_loss)
            logger.add_style_loss(style_loss)
            logger.add_cycle_loss(cycle_loss)

            logger.add_teach_loss(teach_prior_loss, teach_post_loss, kl_obj)

            # ---------------------------------------------------
            real1_loss, real2_loss, fake_loss = self.step_dis(x, x_ins, y_ins, x_id,
                generator, discriminator, None, tool, valid=True)


            logger.add_dis_loss(real1_loss, real2_loss, fake_loss)


        # calculate bleu
        if self.hps.paired_valid_data is not None:
            print ("calculate validation bleu...")
            bleu0to1, bleu1to0 = self.bleu_validator.do_validation(generator, tool)
            metric = "    bleu0to1: %.2f, bleu1to0: %.2f" % (bleu0to1, bleu1to0)
        else:
            metric = ""

        logger.print_log(epoch=epoch, metric=metric)


    #----------------------------------------------------------
    def step_gen(self, x, x_ins, y_ins, y, y_id, generator,
        discriminator, optimizerGen, criterion, tool, valid=False):

        if not valid:
            optimizerGen.zero_grad()

        # y_emb_outs: (B, T, emb_size)
        x_outs, y_emb_outs_prior, y_outs_prior, y_outs_post, kl_obj = \
            generator(x, x_ins, y_ins, y, teacher_forcing=0.9)

        # reconstruct loss
        recon_loss = criterion(x_outs, x[:, 1:], True)

        # supervised loss
        if y is not None:
            teach_prior_loss = criterion(y_outs_prior, y[:, 1:], True)
            teach_post_loss = criterion(y_outs_post, y[:, 1:], True)

            # when paired sentences exist, we give a small teaching ratio
            #   to better stabilize the training
            y_outs_cycle, _ = generator.generate_style_only(x, y_ins, y=y,
                teacher_forcing=0.25,
                with_emb_outs=False, with_outs=True)

            y_seqs = tool.rebuild_outs(y_outs_cycle)
        else:
            teach_prior_loss = torch.zeros_like(recon_loss, device=self.device)
            teach_post_loss = torch.zeros_like(recon_loss, device=self.device)
            kl_obj = torch.zeros_like(recon_loss, device=self.device)
            y_seqs = tool.rebuild_outs(y_outs_prior)


        #cycle loss
        y_seqs = y_seqs.to(self.device)
        y2x_outs, _ = generator.generate_style_only(y_seqs, x_ins,
             with_outs=True, with_emb_outs=False)
        cycle_loss = criterion(y2x_outs, x[:, 1:], True)


        # (B, n_class)
        logits, _ = discriminator(y_emb_outs_prior)

        style_loss = utils.safe_loss(- F.log_softmax(logits, dim=1)[:, y_id])


        superv_loss = (teach_prior_loss + teach_post_loss + 0.2*kl_obj)

        #-------------------------------------------
        loss = recon_loss + style_loss + cycle_loss + superv_loss * self.decay_tool.get_rate()
        loss = loss.mean()

        if (not valid) and (not torch.isnan(loss).item()):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), self.hps.clip_grad_norm)
            optimizerGen.step()

        return recon_loss.mean().item(), style_loss.mean().item(), cycle_loss.mean().item(),\
            teach_prior_loss.mean().item(), teach_post_loss.mean().item(), kl_obj.mean().item(),\
            y_outs_prior, y_outs_post



    def step_dis(self, x, x_ins, y_ins, x_id, generator,
        discriminator, optimizerDis, tool, valid=False):

        with torch.no_grad():
            # (B, L, emb_dim)
            _, y_emb_outs = generator.generate_style_only(x, y_ins)
            _, x_emb_outs = generator.generate_style_only(x, x_ins)
            x_embs = generator.layers['word_embed'](x[:, 1:])

        if not valid:
            optimizerDis.zero_grad()

        # train discriminator
        logits_fake, _ = discriminator(y_emb_outs)

        logits_real1, _ = discriminator(x_emb_outs)
        logits_real2, _ = discriminator(x_embs)


        # M +1-th class indicates a generated fake
        fake_loss = utils.safe_loss(-F.log_softmax(logits_fake, dim=1)[:, 2])

        real_loss1 = utils.safe_loss(-F.log_softmax(logits_real1, dim=1)[:, x_id])
        real_loss2 = utils.safe_loss(-F.log_softmax(logits_real2, dim=1)[:, x_id])

        loss = fake_loss + real_loss1 + real_loss2
        loss = loss.mean()

        if (not valid) and (not torch.isnan(loss).item()):
            loss.backward()
            optimizerDis.step()

        return real_loss1.mean().item(), real_loss2.mean().item(), fake_loss.mean().item()



    def run_train(self, generator, discriminator, tool, optimizerGen, optimizerDis, criterion, logger):

        logger.set_start_time()

        for step in range(0, tool.train_batch_num):

            # (x, x_ins, y_ins, x_id, y_id, y)
            batch = tool.get_train_batch(step)
            x = batch[0].to(self.device)
            x_ins = batch[1].to(self.device)
            y_ins = batch[2].to(self.device)
            x_id, y_id = batch[3], batch[4]

            if batch[5] is not None:
                y = batch[5].to(self.device)
            else:
                y = None

            # -------------------------------------------------------------
            # train generator
            step_recon_loss, step_style_loss, step_cycle_loss = 0.0, 0.0, 0.0
            step_teach_prior_loss, step_teach_post_loss, step_kl_obj = 0.0, 0.0, 0.0

            ngen = self.hps.n_gen
            for k in range(0, ngen):
                recon_loss, style_loss, cycle_loss, teach_prior_loss, teach_post_loss, kl_obj,\
                outs_prior, outs_post = self.step_gen(x, x_ins, y_ins, y, y_id, generator,
                    discriminator, optimizerGen, criterion, tool)

                step_recon_loss += recon_loss
                step_style_loss += style_loss
                step_cycle_loss += cycle_loss

                step_teach_prior_loss += teach_prior_loss
                step_teach_post_loss += teach_post_loss
                step_kl_obj += kl_obj

            logger.add_recon_loss(step_recon_loss / ngen)
            logger.add_style_loss(step_style_loss / ngen)
            logger.add_cycle_loss(step_cycle_loss / ngen)

            logger.add_teach_loss(step_teach_prior_loss / ngen,
                step_teach_post_loss / ngen, step_kl_obj / ngen)

            # ---------------------------------------------------
            # train discriminator
            step_dis_real1_loss, step_dis_real2_loss, step_dis_fake_loss = 0.0, 0.0, 0.0
            ndis = self.hps.n_dis
            for k in range(0, ndis):
                real1_loss, real2_loss, fake_loss = self.step_dis(x, x_ins, y_ins, x_id,
                    generator, discriminator, optimizerDis, tool)

                step_dis_real1_loss += real1_loss
                step_dis_real2_loss += real2_loss
                step_dis_fake_loss += fake_loss

            logger.add_dis_loss(step_dis_real1_loss / ndis,
                    step_dis_real2_loss / ndis, step_dis_fake_loss / ndis)

            logger.set_rate("learning_rate", optimizerGen.rate())
            logger.set_rate("teach_ratio", self.decay_tool.get_rate())
            self.decay_tool.do_step()

            # ---------------------------------------------------
            if step % self.hps.log_steps == 0:
                logger.set_end_time()
                utils.sample(x, outs_prior, outs_post, y, x_id, y_id, self.hps.sample_num, tool)
                logger.print_log()
                logger.set_start_time()



    def train(self, generator, discriminator, tool):
        print ("using device: %s" %(self.device))
        '''
        print ("discriminator parameters!")
        utils.print_parameter_list(discriminator)
        print ("______________________")
        input(">")
        '''

        # build training and validation data
        tool.close_corruption()
        tool.set_batch_size(self.hps.batch_size)
        tool.build_train_data(self.hps.unpaired_train_data, self.hps.paired_train_data, None)
        tool.build_valid_data(self.hps.unpaired_valid_data, self.hps.paired_valid_data, None)

        print ("train batch num: %d" % (tool.train_batch_num))
        print ("valid batch num: %d" % (tool.valid_batch_num))

        # training logger
        logger = StyInsLogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.log_steps)
        logger.set_log_path(self.hps.train_log_path)
        logger.set_rate('learning_rate', 0.0)
        logger.set_rate('teach_ratio', 1.0)


        # build optimizer
        optGen = torch.optim.Adam(generator.parameters(),
            lr=0.0, betas=(0.9, 0.999))
        optDis = torch.optim.Adam(discriminator.parameters(),
            lr=0.0, betas=(0.5, 0.999))

        optimizerGen = ISRScheduler(optimizer=optGen, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr, init_lr=self.hps.init_lr, beta=0.75)

        optimizerDis = ISRScheduler(optimizer=optDis, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr, init_lr=self.hps.init_lr, beta=0.5)

        # ----------------------------------------------
        criterion = Criterion(self.hps.pad_idx)

        # when paired data is available, at the beginning,
        #   we give larger weight for supervised loss and then decrease it.
        self.decay_tool = LinearDecay(burn_down_steps=self.hps.superv_burn_down_steps,
            decay_steps=self.hps.superv_decay_steps, max_v=self.hps.superv_max, min_v=self.hps.superv_min)


        generator.train()
        discriminator.train()

        print ("In training mode: %d, %d" % (generator.training, discriminator.training))
        #input("Please check the parameters and press enter to continue>")

        # --------------------------------------------------------------------------------------
        for epoch in range(1, self.hps.max_epoches+1):

            self.run_train(generator, discriminator, tool,
                optimizerGen, optimizerDis, criterion, logger)

            if (self.hps.save_epoches >= 1) and (epoch % self.hps.save_epoches) == 0:
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint_multiple(self.hps.ckpt_path,
                    generator, discriminator, optimizerGen, optimizerDis, epoch)

            if epoch % self.hps.valid_epoches == 0:
                print("run validation...")
                generator.eval()
                discriminator.eval()
                print ("in training mode: %d, %d" % (generator.training, discriminator.training))
                self.run_validation(epoch, generator, discriminator, criterion, tool, optimizerGen.rate())
                generator.train()
                discriminator.train()
                print ("validation Done, mode: %d, %d" % (generator.training, discriminator.training))

            logger.add_epoch()

            # each epoch, we rebuild the data, including shuffling the data, re-assigning
            #   transfer direction and resampling instances for each batch
            print("shuffle data...")
            tool.shuffle_training_data()


def main():
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main()
