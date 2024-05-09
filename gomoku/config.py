import os
import copy
import random
import time
from operator import itemgetter
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import gym
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
from IPython import display

board_width = 6
board_height = 6
n_in_row = 4
c_puct = 5
n_playout = 100 # 每一步进行mcts搜索时，模拟的次数
learn_rate = 0.002
lr_multiplier = 1.0
temperature = 1.0
noise_eps = 0.75
dirichlet_alpha = 0.3
buffer_size = 5000
train_batch_size = 128
update_epochs = 5
kl_coeff = 0.02
checkpoint_freq = 20
mcts_infer = 200 # 纯mcts推理时间
restore_model = None
game_batch_num = 40
model_path = './model_path/'
