import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import gym

GAMMA = 0.9


# Policy gradient在处理连续动作时，输出是均值和方差的分布参数，然后使用这些参数来采样连续动作
class PolicyNetwork(nn.Module):
    def __init__(self, n_features, n_actions) -> None:
        super(PolicyNetwork, self).__init__()
        self.L1 = nn.Linear(n_features, 30)
        self.L2 = nn.Linear(30, 30)
        self.mean = nn.Linear(30, n_actions)
        self.log_std = nn.Linear(30, n_actions)

    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std


class Actor:
    def __init__(self, n_features, n_actions, lr=0.001) -> None:
        self.actor_net = PolicyNetwork(n_features, n_actions)
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)

    def learn(self, state, action, td_error):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32)
        td_error = torch.tensor(td_error, dtype=torch.float32)

        mean, log_std = self.actor_net(s)
        std = torch.exp(log_std)
        normal = distributions.Normal(mean, std)

        log_prob = normal.log_prob(a)
        exp_v = torch.mean(log_prob * td_error)

        self.optimizer.zero_grad()
        (-exp_v).backward()
        self.optimizer.step()

        return exp_v.item()

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        mean, log_std = self.actor_net(s)
        std = torch.exp(log_std)
        normal = distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action)
        action = torch.tanh(action)
        action = action.detach().numpy()

        return action


class CriticNetwork(nn.Module):
    def __init__(self, n_features, n_actions, **kwargs) -> None:
        super(CriticNetwork, self).__init__(n_features, **kwargs)
        self.l1 = nn.Linear(n_features, 30)
        self.l2 = nn.Linear(30, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)

        return x


class Critic:
    def __init__(self, n_features, lr=0.01) -> None:
        self.critic_net = CriticNetwork(n_features, 1)
        self.optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)

    def learn(self, state, reward, state_):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        r = torch.tensor(reward, dtype=torch.float32)
        s_ = torch.tensor(state_, dtype=torch.float32).unsqueeze(0)

        v = self.critic_net(s)
        v_ = self.critic_net(s_)

        td_error = r + GAMMA * v_ - v
        loss = torch.square(td_error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error.item()


MAX_EPISODE = 1000
DISPLAY_REWARD_THERSHOLD = 200
RENDER = False

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n
print('action', N_A)

actor = Actor(n_features=N_F, n_actions=N_A)
critic = Critic(n_features=N_F)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER:
            env.render()

        a = actor.choose_action(s)
        s_, r, done, info, _ = env.step(a)

        if done: r = -20
        track_r.append(r)

        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1

        if done or t >= MAX_EPISODE:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

            if running_reward > DISPLAY_REWARD_THERSHOLD: RENDER = True
            print("episode:", i_episode, "  reward:", int(running_reward))
            break