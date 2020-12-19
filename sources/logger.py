# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-12-19 10:09:42
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
'''
import numpy as np
import time

class InfoLogger(object):
    """docstring for LogInfo"""
    def __init__(self, mode):
        super(InfoLogger).__init__()
        self._mode = mode # string, 'train' or 'valid'
        self._total_steps = 0
        self._batch_num = 0
        self._log_steps = 0
        self._cur_step = 0
        self._cur_epoch = 1

        self._start_time = 0
        self._end_time = 0

        #--------------------------
        self._log_path = "" # path to save the log file

        # -------------------------
        self._decay_rates = {'learning_rate':1.0,
            'teach_ratio':1.0, 'temperature':1.0}


    def set_batch_num(self, batch_num):
        self._batch_num = batch_num
    def set_log_steps(self, log_steps):
        self._log_steps = log_steps
    def set_log_path(self, log_path):
        self._log_path = log_path

    def set_rate(self, name, value):
        self._decay_rates[name] = value


    def set_start_time(self):
        self._start_time = time.time()

    def set_end_time(self):
        self._end_time = time.time()

    def add_step(self):
        self._total_steps += 1
        self._cur_step += 1

    def add_epoch(self):
        self._cur_step = 0
        self._cur_epoch += 1


    # ------------------------------
    @property
    def cur_process(self):
        ratio = float(self._cur_step) / self._batch_num * 100
        process_str = "%d/%d %.1f%%" % (self._cur_step, self._batch_num, ratio)
        return process_str

    @property
    def time_cost(self):
        return (self._end_time-self._start_time) / self._log_steps

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def epoch(self):
        return self._cur_epoch

    @property
    def mode(self):
        return self._mode

    @property
    def log_path(self):
        return self._log_path


    @property
    def learning_rate(self):
        return self._decay_rates['learning_rate']

    @property
    def teach_ratio(self):
        return self._decay_rates['teach_ratio']

    @property
    def temperature(self):
        return self._decay_rates['temperature']


#------------------------------------
class StyInsLogger(InfoLogger):
    def __init__(self, mode):
        super(StyInsLogger, self).__init__(mode)
        self._gen_loss = 0.0

        self._total_recon_loss = []
        self._total_style_loss = []
        self._total_cycle_loss = []

        self._total_teach_prior_loss = []
        self._total_teach_post_loss = []
        self._total_teach_kl_obj = []

        self._total_dis_fake_loss = []
        self._total_dis_real1_loss = []
        self._total_dis_real2_loss = []


    def add_recon_loss(self, loss):
        self._total_recon_loss.append(loss)
        self.add_step()

    def add_style_loss(self, loss):
        self._total_style_loss.append(loss)

    def add_cycle_loss(self, loss):
        self._total_cycle_loss.append(loss)

    def add_teach_loss(self, prior_loss, post_loss, kl_obj):
        if prior_loss > 0 and post_loss > 0:
            self._total_teach_prior_loss.append(prior_loss)
            self._total_teach_post_loss.append(post_loss)
            self._total_teach_kl_obj.append(kl_obj)


    def add_dis_loss(self, real1_loss, real2_loss, fake_loss):
        self._total_dis_real1_loss.append(real1_loss)
        self._total_dis_real2_loss.append(real2_loss)
        self._total_dis_fake_loss.append(fake_loss)

    def get_total_teach_steps(self):
        return len(self._total_teach_prior_loss)


    def print_log(self, epoch=None, metric=None):

        def get_loss(vec):
            if len(vec) == 0:
                return 0.0
            else:
                return np.mean(vec)

        recon_loss = get_loss(self._total_recon_loss)
        ppl = np.exp(recon_loss)

        style_loss = get_loss(self._total_style_loss)

        cycle_loss = get_loss(self._total_cycle_loss)

        dis_real1_loss = get_loss(self._total_dis_real1_loss)
        dis_real2_loss = get_loss(self._total_dis_real2_loss)

        dis_fake_loss = get_loss(self._total_dis_fake_loss)
        #---------------------
        #
        teach_prior_loss = get_loss(self._total_teach_prior_loss)
        teach_post_loss = get_loss(self._total_teach_post_loss)
        teach_kl_obj = get_loss(self._total_teach_kl_obj)

        teach_steps = self.get_total_teach_steps()

        if self.mode == 'train':
            process_info = "epoch: %d, %s, %.2fs per iter, lr: %.4f, tr: %.2f, ts: %d" % (self.epoch,
                self.cur_process, self.time_cost, self.learning_rate, self.teach_ratio, teach_steps)
        else:
            process_info = "epoch: %d, lr: %.4f, tr: %.2f" % (
                epoch, self.learning_rate, self.teach_ratio)

        # -------------------------------------------
        train_info1 = "    recon loss: %.3f  ppl:%.2f, style loss: %.3f, cycle loss: %.3f" \
            % (recon_loss, ppl, style_loss, cycle_loss)

        train_info2 = "    dis real1 loss: %.3f, dis real2 loss: %.3f, dis fake loss: %.3f" \
            % (dis_real1_loss, dis_real2_loss, dis_fake_loss)

        train_info3 = "    teach prior loss: %.3f, teach post loss: %.3f, teach kl obj: %.3f" \
            % (teach_prior_loss, teach_post_loss, teach_kl_obj)


        info = process_info + "\n" + train_info1 + "\n" + train_info2 + "\n" + train_info3
        print (info)
        if self.mode == 'valid':
            print (metric)
        print ("______________________")

        if self.mode == 'train':
            info_str = info + "\n\n"
        else:
            info_str = info + "\n" + metric + "\n\n"

        fout = open(self.log_path, 'a')
        fout.write(info_str)
        fout.close()



# ----------------------------------------------
class PreLogger(InfoLogger):
    def __init__(self, mode):
        super(PreLogger, self).__init__(mode)

        self._total_recon_loss = 0.0


    def add_recon_loss(self, loss):
        self._total_recon_loss += loss
        self.add_step()


    def print_log(self, epoch=None):

        recon_loss = self._total_recon_loss / self.total_steps
        ppl = np.exp(recon_loss)


        if self.mode == 'train':
            process_info = "epoch: %d, %s, %.2fs per iter, lr: %.4f" % (self.epoch,
                self.cur_process, self.time_cost, self.learning_rate)
        else:
            process_info = "epoch: %d, lr: %.4f" % (epoch, self.learning_rate)

        # ---------------------------------------------------
        train_info = "    recon loss: %.3f  ppl:%.2f" % (recon_loss, ppl)

        print (process_info)
        print (train_info)
        print ("______________________")

        info_str = process_info + "\n" + train_info + "\n\n"

        fout = open(self.log_path, 'a')
        fout.write(info_str)
        fout.close()