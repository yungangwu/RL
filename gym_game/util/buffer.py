import random
import torch
import numpy as np

from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size) -> None:
        self.buffer = deque(maxlen=buffer_size)

    def push(self, play_data):
        '''
        play_data: [(state, act, winner_z, state_), ..., ...]
        '''
        self.buffer.extend(play_data)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, acts, winner_zs, states_ = zip(*experiences)

        # 将这些部分分别转换成批次
        states_batch = np.stack(states).astype(np.float32)
        acts_batch = np.stack(acts).astype(np.float32)
        winner_zs_batch = np.stack(winner_zs).astype(np.float32)
        states_batch_ = np.stack(states_).astype(np.float32)

        return states_batch, acts_batch, winner_zs_batch, states_batch_

    def len(self): # 如果类内不定义该方法，则不能通过内置方法len直接获取对象的长度
        return len(self.buffer)