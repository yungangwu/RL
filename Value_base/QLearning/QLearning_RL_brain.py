import numpy as np
import pandas as pd
from util.RL import RL

class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exits(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'teminal':
            # q_target是当前获得的奖励加上下一个状态s_所能获得的最大奖励，以及一个衰减过程（未来奖励）
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

