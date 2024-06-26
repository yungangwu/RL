import random
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from train.model import QNetwork
from util.buffer import ReplayBuffer
from util.common import Player
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class RandomAgent:
    def __init__(self, env, agent_name):
        self.env = env
        self.agent_name = agent_name

    def act(self, state, legal_moves):
        '''
        返回随机位置作为当前玩家的落子位置。后续改为用模型确定落子位置。
        '''
        action = np.random.choice(np.flatnonzero(legal_moves))
        return action


class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 agent_name,
                 buffer_size=2000,
                 batch_size=32,
                 gamma=0.95,
                 lr=0.001,
                 eps=0.9):
        self.epsilon = eps
        self.state_size = state_size
        self.action_size = action_size
        self.agent_name = agent_name
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.loss = None

        self.epsilon_start = eps
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.99


        self.local_network = QNetwork(action_size)
        self.target_network = QNetwork(action_size)
        self.optimizer = optim.Adam(self.local_network.parameters(),
                                    lr=self.lr)

        self.memory = ReplayBuffer(buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, legal_moves):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.local_network(state).squeeze().detach().numpy()
        q_values[~legal_moves] = -np.inf
        if np.random.uniform() < self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.choice(np.flatnonzero(legal_moves))

        return action

    def train(self, epochs, save_epoch):
        while True:
            if len(self.memory) > self.batch_size:
                break

        for epoch in epochs:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences, self.gamma)

            if epoch % save_epoch == 0:
                save_model_path = '/home/yg/code/ReinforcementLearning/wuziqi/path_to_model/' + f'episode_{epoch}'
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)
                model_path = save_model_path + '/model.pth'
                self.save_model(model_path)

    def update_target_network(self,):
        self.target_network.load_state_dict(self.local_network.state_dict())

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        q_values = self.local_network(states).gather(1, actions.unsqueeze(1))

        next_q_values = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        self.loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.update_target_network()

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_end, self.epsilon)

    def update_target_network(self):
        self.target_network.load_state_dict(self.local_network.state_dict())

    def load_model(self, model_path):
        self.local_network.load_state_dict(torch.load(model_path))
        self.target_network.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        torch.save(self.local_network.state_dict(), model_path)
