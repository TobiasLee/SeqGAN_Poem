from mydis import Discriminator
from mygen import Generator
from dataloader import Gen_Data_loader, Dis_dataloader
import random
import numpy as np
import tensorflow as tf
from myG_beta import G_beta


dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75

EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 128  # hidden state dimension of lstm cell
SEQ_LENGTH = 20  # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 1  # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
vocab_size = 6915 # max idx of word token = 6914

dis_emb_size = 64

TOTAL_BATCH = 20
positive_file = './train.txt'
negative_file = './generator_sample.txt'
eval_file = './eval_file.txt'
generated_num = 1000
sample_time = 16 # for G_beta to get reward
num_class = 2 # 0 : fake data 1 : real data

G = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
G_beta = G_beta(G, 0.8)