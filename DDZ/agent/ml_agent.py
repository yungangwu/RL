import torch
import torch.nn as nn
import numpy as np

from modle import Net
from util import Encode, LSTMEncode


# 定义几个编码，手牌编码，打牌记录编码(LSTM)，上一个玩家的动作，及其他玩家手牌数量的信息，地主信息，炸弹数量，编码
class DeepQNetwork:
    def __init__(self, n_actions,
                       n_features,
                       learning_rate=0.01,
                       reward_decay=0.9,
                       e_greedy=0.9,
                       replace_target_iter=200,
                       memory_size=2000,
                       batch_size=64,
                ):
        self.n_actions = n_actions
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy = e_greedy

        self.learn_step_counter = 0

        self.encode_instance = Encode()

        self.eval_net = Net(n_actions, n_features)
        self.target_net = Net(n_actions, n_features)

        self.replace_target_iter = replace_target_iter
        self.memory_counter = 0
        self.memory = np.zeros((memory_size, n_features * 2 + 2)) # memory中要存两obs，一个action和一个reward，所以尺寸是obs*2+2
        self.optimizer = torch.optim.Adam(self.eval_net.paramters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.cost_his = []

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0) # 用于在指定维度上插入一个大小为1的维度

        if np.random.uniform() < self.e_greedy:
            action_values = self.eval_net.forward(x)
            action = torch.max(action_values, 1)[1].data.numpy()[0, 0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transiton(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) # 用于水平（按列）拼接多个数组
        # 如果记忆库满了就覆盖老数据
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_features])
        b_a = torch.LongTensor(b_memory[:, self.n_features:self.n_features+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_features+1:self.n_features+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_features:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_target_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_target_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)
        self.cost_his.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def encode_state(self,):
        pass
    