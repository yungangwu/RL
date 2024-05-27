import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config.config import *
from model.base_policy_value import BasePolicyValue

class DQN(nn.Module):
    def __init__(self, board_size):
        super(DQN, self).__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * board_size * board_size, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = self.fc(x)
        x = x.view(-1, self.board_size * self.board_size)
        return x

class DQNPolicyValue(BasePolicyValue):
    def __init__(self, model_file, epsilon=0.8) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.l2_const = 1e-4
        self.gamma = 0.9
        self.epsilon = epsilon

        self.policy_value_net = DQN(board_size=board_width).to(self.device)
        self.target_policy_value_net = DQN(board_size=board_width).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)
        self.loss_func = nn.MSELoss()

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def get_move(self, sensible_moves, state):
        # print('sensible_moves', sensible_moves, 'state \n', state)
        state = torch.from_numpy(state.copy()).to(self.device)

        q_values = self.policy_value_net(state).cpu().squeeze().detach().numpy()

        # 创建一个布尔数组，标记哪些动作是合法的
        sensible_moves_np = np.isin(np.arange(q_values.size), sensible_moves)

        # 将不合法动作的Q值设置为负无穷
        q_values[~sensible_moves_np] = -np.inf

        if np.random.uniform() < self.epsilon:
            move = np.argmax(q_values)
        else:
            move = np.random.choice(np.flatnonzero(sensible_moves))

        return move

    def train_step(self, meta_data, lr):
        state_batch, act_batch, winner_batch, state_batch_ = meta_data
        state_batch = state_batch.to(self.device)
        act_batch = act_batch.to(self.device)
        winner_batch = winner_batch.to(self.device)
        state_batch_ = state_batch_.to(self.device)

        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # 获取 act_batch 为 1 的索引
        _, act_indices = act_batch.max(dim=1)

        # 使用 gather 方法获取每行中指定位置的 q_value
        q_value = self.policy_value_net(state_batch).gather(1, act_indices.unsqueeze(1)).squeeze()

        q_target_next = self.target_policy_value_net(state_batch_).detach()
        q_target = winner_batch + self.gamma * q_target_next.max(1)[0]
        loss = self.loss_func(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)
