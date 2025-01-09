import random
import torch
import numpy as np

from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size) -> None:
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward):
        self.buffer.extend(zip(state, action, reward))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        # states = torch.Tensor([exp[0] for exp in experiences])
        # actions = torch.Tensor([exp[1] for exp in experiences]).long()
        # rewards = torch.Tensor([exp[2] for exp in experiences])
        # next_states = torch.Tensor([exp[3] for exp in experiences])
        # dones = torch.Tensor([exp[4] for exp in experiences])
        states, actions, rewards = zip(*experiences)

        return states, actions, rewards

    def clear(self):
        self.buffer.clear()

    def size(self): # 如果类内不定义该方法，则不能通过内置方法len直接获取对象的长度
        return len(self.buffer)