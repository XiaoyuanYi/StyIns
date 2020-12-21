from collections import namedtuple
import torch

HParams = namedtuple('HParams',
   'vocab_size, pad_idx, bos_idx,'
   'emb_size, hidden_size, flow_h_size, made_size, flow_depth,'
   'n_ins, n_dis, n_gen, r_superv,'
   'drop_ratio, attn_drop_ratio, weight_decay, clip_grad_norm,'
   'warmup_steps, max_lr, min_lr, init_lr,'
   'superv_max, superv_min, superv_burn_down_steps, superv_decay_steps,'
   'infor_nats, infor_groups,'
   'max_len, sample_num, batch_size, max_epoches,'
   'save_epoches, valid_epoches, log_steps,'
   'vocab_path, unpaired_train_data, unpaired_valid_data,'
   'pretrain_method, corrupt_ratio, pre_epoches, pre_batch_size,'
   'pre_log_steps, pre_valid_epoches, pre_save_epoches,'
   'pre_train_log_path, pre_valid_log_path,'
   'paired_train_data, paired_valid_data, ckpt_path, train_log_path, valid_log_path,'
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



yelp_hps = HParams(
   vocab_size=-1, pad_idx=-1, bos_idx=-1, # to be replaced by true indices after loading dictionary.
   emb_size=256, hidden_size=512,
   n_dis=1, n_gen=1, # in each iteration, train the two modules n_dis, n_gen times respective
   flow_h_size=512, made_size=512, flow_depth=6, # hyper-parameters for the IAF flow
   n_ins=10, # the number of style instances

   drop_ratio=0.1, attn_drop_ratio=0.1, weight_decay=5e-5, clip_grad_norm=2.0,
   warmup_steps=6000, max_lr=8e-4, min_lr=5e-5, init_lr=1e-5,

   # settings for semi-supervised training
   r_superv=0.0, # the ratior of paired data to be used for semi-supervised training
   superv_max=0.75, superv_min=0.1, # the weight teching loss
   superv_burn_down_steps=400, superv_decay_steps=1500,
   infor_nats=0.125, infor_groups=128, # hyper-parameters for the kl objective

   # sentences that are longger than max_len will be removed
   batch_size=128, max_epoches=12, max_len=20, sample_num=1,
   save_epoches=2, valid_epoches=1, log_steps=200,

   train_log_path="../log/yelp_train_log.txt",
   valid_log_path="../log/yelp_valid_log.txt",

   #-----------------------------
   # settings for pre-training
   # we provide two methods for pretraining:
   #
   # lm: pretrain a language model whose parameters will be used to initialize
   #    the encoders and deocder. This method is utilized in our paper which can
   #    achieve better performance but is sometimes unstable
   # dae: pretrain the source encoder, style encoder and decoder as a denoising
   #    autoencoder. We corrupt the source sentence by randomly replacing some tokens
   #    with UNK so as to for the style encoder to capture more stylistic information to
   #    help reconstruct the sentence. This method is more stable and we provide it as an
   #    alternate for users.
   pretrain_method='lm', # lm or dae
   corrupt_ratio=0.1, # for dae only
   # pre_epoches=12 for lm pretraining and 2 or 4 for dae pretraining (also for gyafc)
   pre_epoches=12, pre_batch_size=128,
   pre_log_steps=200, pre_valid_epoches=1, pre_save_epoches=2,
   pre_train_log_path="../log/yelp_pre_train_log.txt",
   pre_valid_log_path="../log/yelp_pre_valid_log.txt",
   # ----------------------------

   vocab_path="../corpus/yelp_vocab.txt",

   unpaired_train_data="../corpus/yelp_unpaired_train.json",
   unpaired_valid_data="../corpus/yelp_unpaired_valid.json",

   paired_train_data=None,
   paired_valid_data=None,

   ckpt_path="../ckpts/" # the path to save checkpoints.
)


# ----------------------------------------------------

gyafc_hps = HParams(
   vocab_size=-1, pad_idx=-1, bos_idx=-1, # to be replaced by true indices after loading dictionary.
   emb_size=256, hidden_size=512, n_dis=5, n_gen=1, # in each iteration, train the discriminator n_dis times
   flow_h_size=512, made_size=512, flow_depth=6, # hyper-parameters for the IAF flow
   n_ins=10, # the number of instances


   drop_ratio=0.25, attn_drop_ratio=0.1, weight_decay=5e-5, clip_grad_norm=2.0,
   warmup_steps=3000, max_lr=8e-4, min_lr=5e-5, init_lr=1e-5,

   r_superv=0.24, # the ratior of paired data to be used for semi-supervised training
   superv_max=0.75, superv_min=0.1,
   superv_burn_down_steps=2000, superv_decay_steps=4000,
   infor_nats=0.125, infor_groups=128,

   batch_size=64, max_epoches=20, max_len=32, sample_num=1,
   save_epoches=2, valid_epoches=1, log_steps=100,

   train_log_path="../log/gyafc_train_log.txt",
   valid_log_path="../log/gyafc_valid_log.txt",

   #-----------------------------
   # for pre-training
   pretrain_method='lm', # lm or dae
   corrupt_ratio=0.1,
   pre_epoches=12, pre_batch_size=128,
   pre_log_steps=200, pre_valid_epoches=1, pre_save_epoches=2,
   pre_train_log_path="../log/gyafc_pre_train_log.txt",
   pre_valid_log_path="../log/gyafc_pre_valid_log.txt",
   # ----------------------------

   vocab_path="../corpus/gyafc_vocab.txt",

   unpaired_train_data="../corpus/gyafc_unpaired_train.json",
   unpaired_valid_data="../corpus/gyafc_unpaired_valid.json",

   paired_train_data="../corpus/gyafc_paired_train.json",
   paired_valid_data="../corpus/gyafc_paired_valid.json",

   ckpt_path="../ckpts/" # the path to save checkpoints.

)
