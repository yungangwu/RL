import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from util.buffer import ReplayBuffer
from torch.distributions import Categorical

# TODO:
# 2、训练模型要测试其强度，赢了才能保存
# 3、将best模型向各个环境进行推送


class ACNet(nn.Module):
    def __init__(self, height, wide, action_dim) -> None:
        super(ACNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        conv_output_size = self._get_conv_output((3, height, wide))

        self.fc = nn.Linear(conv_output_size, action_dim)
        self.value_fc = nn.Linear(conv_output_size, 1)

    def _get_conv_output(self, shape):
        o = self.pool(self.conv2(self.pool(self.conv1(torch.zeros(1, *shape)))))
        return int(np.prod(o.size())) # prod计算所有元素的乘积

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.reshape(x.size(0), -1)

        action_probs = torch.softmax(self.fc(x), dim=-1)
        state_value = self.value_fc(x)
        return action_probs, state_value

A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10

class PPOPolicyValue:
    def __init__(self, height, wide, action_dim, epsilon=0.8) -> None:

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.pi_action_net = ACNet(height, wide, action_dim).to(self.device)
        self.old_action_net = ACNet(height, wide, action_dim).to(self.device)

        self.training_steps = 0
        self.epsilon = epsilon
        self.optimpi = optim.Adam(self.pi_action_net.parameters())

    def get_action(self, state, action_space):
        state = torch.tensor(state.copy()).float().to(self.device)
        state = state.permute(2, 0, 1).unsqueeze(0)
        action_probs, _ = self.pi_action_net(state)
        action_probs_cpu = action_probs.cpu().detach().squeeze()
        binary_actions = torch.zeros_like(action_probs_cpu)

        # 选择概率最高的
        if np.random.uniform() < self.epsilon:
            move_prob, move_index = torch.max(action_probs_cpu, 0)
            move = move_index.item()
        else:
            move = np.random.choice(action_space)

        binary_actions[move] = 1.0
        # print('binary_actions', binary_actions)

        return binary_actions

    def get_value(self, state):
        state = torch.tensor(state.copy()).float().to(self.device)
        state = state.permute(2, 0, 1).unsqueeze(0)
        _, value = self.pi_action_net(state)

        return value

    def train_step(self, buffer: ReplayBuffer, train_batch_size = 128):
        # print('ppo train' + '*'*20)
        state_batch, act_batch, winner_batch, state_batch_ = buffer.sample(train_batch_size)
        # print('state_batch', type(state_batch), state_batch.shape)

        state_batch = torch.tensor(state_batch).to(self.device)
        act_batch = torch.tensor(act_batch).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)
        state_batch_ = torch.tensor(state_batch_).to(self.device)

        self.update_old_ac()

        for _ in range(A_UPDATE_STEPS):
            act_loss, kl = self.comput_actor_loss(state_batch, act_batch, winner_batch)
            if kl > 4 * 0.01:
                break

        for _ in range(C_UPDATE_STEPS):
            self.comput_critic_loss(state_batch, winner_batch)

        self.training_steps += 1
        torch.cuda.empty_cache()

        if self.training_steps % 200 == 0:
            self.save_model(f'/home/yg/code/test/ReinforceLearning/gym_game/modle_file/model_epoch_{self.training_steps}.pt')

        return act_loss

    def comput_actor_loss(self, state_batch, action_batch, reward_batch):
        pi_act_probs, v = self.pi_action_net(state_batch)
        td_error = reward_batch - v

        pi_action_probs = torch.sum(pi_act_probs * action_batch, dim=1)
        pi_m = Categorical(probs=pi_act_probs)

        old_act_probs, _ = self.old_action_net(state_batch)
        old_action_probs = torch.sum(old_act_probs * action_batch, dim=1)
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

    def get_policy_param(self,):
        return self.pi_action_net.state_dict()

    def save_model(self, model_file) -> str:
        print('save model', '*'*20)
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)

    def update_old_ac(self,):
        self.old_action_net.load_state_dict(self.pi_action_net.state_dict())

    def comput_critic_loss(self, states, rewards):
        values = self._loss_critic(states, rewards)
        loss = F.mse_loss(values, rewards)

        self.optimpi.zero_grad()
        loss.backward()
        self.optimpi.step()

        return loss.item()

    def _loss_critic(self, states, rewards):
        _, values = self.pi_action_net(states)
        values = values.squeeze()
        td_error = rewards - values
        c_loss = td_error.pow(2)
        return c_loss