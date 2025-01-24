import random
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size) -> None:
        self.buffer = deque(maxlen=buffer_size)

    def push(self, obs, action, reward):
        self.buffer.extend(zip(obs, action, reward))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        obses, actions, rewards = zip(*experiences)

        return obses, actions, rewards

    def clear(self):
        self.buffer.clear()

    def size(self): # 如果类内不定义该方法，则不能通过内置方法len直接获取对象的长度
        return len(self.buffer)