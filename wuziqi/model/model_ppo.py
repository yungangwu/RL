import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from config.config import *
from model.base_policy_value import BasePolicyValue
from torch.distributions import Categorical

class ACNet(nn.Module):
    def __init__(self, board_size, a_dim):
        super(ACNet, self).__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.a = nn.Linear(128 * board_size * board_size, a_dim)

        self.c1 = nn.Linear(128 * board_size * board_size, 100)
        self.v = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 128* self.board_size * self.board_size)
        a = F.log_softmax(self.a(x), dim=1)

        c = F.relu(self.c1(x))
        v = self.v(c)
        return a, v

A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10

class PPOPolicyValue(BasePolicyValue):
    def __init__(self, model_file, a_dim, epsilon=0.8) -> None:
        super(PPOPolicyValue, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pi_acnet = ACNet(board_width, a_dim).to(self.device)
        self.old_acnet = ACNet(board_width, a_dim).to(self.device)
        self.optimpi = optim.Adam(self.pi_acnet.parameters())

        self.epsilon = epsilon

        if model_file:
            net_params = torch.load(model_file)
            self.pi_acnet.load_state_dict(net_params)

    def get_move(self, sensible_moves, state):
        state = torch.from_numpy(state.copy()).to(self.device)
        action_probs, value = self.pi_acnet(state)
        action_probs = action_probs.cpu().squeeze().detach().numpy()

        sensible_moves_np = np.isin(np.arange(len(action_probs)), sensible_moves)
        action_probs[~sensible_moves_np] = -np.inf

        # 选择概率最高的
        if np.random.uniform() < self.epsilon:
            move = np.argmax(action_probs)
        else:
            move = np.random.choice(np.flatnonzero(sensible_moves))

        # # 通过随机的方式选择动作
        # mask = torch.ones_like(action_probs) * -1e8
        # mask[sensible_moves] = 0
        # masked_logits = action_probs + mask

        # probs = torch.softmax(masked_logits, dim=-1)
        # m = Categorical(probs)
        # action = m.sample()
        # move = action.item()

        return move

    def train_step(self, meta_data, lr):
        state_batch, act_batch, winner_batch, state_batch_ = meta_data
        state_batch = state_batch.to(self.device)
        act_batch = act_batch.to(self.device)
        winner_batch = winner_batch.to(self.device)
        state_batch_ = state_batch_.to(self.device)

        self.update_old_ac()

        for _ in range(A_UPDATE_STEPS):
            act_loss, kl = self.comput_actor_loss(state_batch, act_batch, winner_batch)
            if kl > 4 * 0.01:
                break

        for _ in range(C_UPDATE_STEPS):
            self.comput_critic_loss(state_batch, winner_batch)

        return act_loss

    def get_policy_param(self,):
        return self.pi_acnet.state_dict()

    def save_model(self, model_file) -> str:
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)

    def update_old_ac(self,):
        self.old_acnet.load_state_dict(self.pi_acnet.state_dict())

    def comput_actor_loss(self, state_batch, act_batch, winner_batch):
        pi_act_probs, v = self.pi_acnet(state_batch)
        td_error = winner_batch - v
        # 使用矩阵乘法从动作概率分布中选择对应于每个动作的概率
        pi_action_probs = torch.sum(pi_act_probs * act_batch, dim=1)
        pi_m = Categorical(probs=pi_act_probs)

        old_act_probs, _ = self.old_acnet(state_batch)
        old_action_probs = torch.sum(old_act_probs * act_batch, dim=1)
        old_m = Categorical(probs=old_act_probs)
        ratio = pi_action_probs - old_action_probs
        surr = ratio * td_error

        kl = torch.distributions.kl_divergence(old_m, pi_m)
        kl_mean = torch.mean(kl)
        loss = -torch.mean(surr - 0.9 * kl)

        self.optimpi.zero_grad()
        loss.backward()
        self.optimpi.step()

        return loss.item(), kl_mean.item()

    def comput_critic_loss(self, states, rewards):
        values = self._loss_critic(states, rewards)
        loss = F.mse_loss(values, rewards)

        self.optimpi.zero_grad()
        loss.backward()
        self.optimpi.step()

        return loss.item()

    def _loss_critic(self, states, rewards):
        _, values = self.pi_acnet(states)
        values = values.squeeze()
        td_error = rewards - values
        c_loss = td_error.pow(2)
        return c_loss
