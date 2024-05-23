import random
import torch
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

        states = torch.Tensor([exp[0] for exp in experiences])
        actions = torch.Tensor([exp[1] for exp in experiences]).long()
        winner_batch = torch.Tensor([exp[2] for exp in experiences])
        states_ = torch.Tensor([exp[3] for exp in experiences])

        return states, actions, winner_batch, states_

    def __len__(self): # 如果类内不定义该方法，则不能通过内置方法len直接获取对象的长度
        return len(self.buffer)