import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import gym
import time

ENV_NAME = 'Pendulum-v0'
MEMORY_CAPACITY = 1000
BATCH_SIZE = 64
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
MAX_EPISODES = 200
MAX_EP_STEPS = 200
RENDER = False


class ActorNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound, **kwargs) -> None:
        super(ActorNetwork, self).__init__(**kwargs)

        self.l1 = nn.Linear(s_dim, 30)
        self.l2 = nn.Linear(30, a_dim)
        self.a_bound = a_bound

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.tanh(self.l2(x))

        return x * self.a_bound


class CriticNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, **kwargs) -> None:
        super(CriticNetwork, self).__init__(**kwargs)
        self.l1 = nn.Linear(s_dim + a_dim, 30)
        self.l2 = nn.Linear(30, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.l1(x))
        x = self.l2(x)

        return x


class DDPG:
    def __init__(self, s_dim, a_dim, a_bound) -> None:
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1),
                               dtype=np.float32)
        self.pointer = 0

        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound

        self.actor_eval = ActorNetwork(s_dim, a_dim, a_bound)
        self.actor_target = ActorNetwork(s_dim, a_dim, a_bound)
        self.critic_eval = CriticNetwork(s_dim, a_dim)
        self.critic_target = CriticNetwork(s_dim, a_dim)

        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(),
                                          lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(),
                                           lr=LR_C)

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        return self.actor_eval(s).detach().numpy()[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.tensor(bt[:, :self.s_dim], dtype=torch.float32)
        ba = torch.tensor(bt[:, self.s_dim:self.s_dim + self.a_dim],
                          dtype=torch.float32)
        br = torch.tensor(bt[:, -self.s_dim - 1:-self.s_dim],
                          dtype=torch.float32)
        bs_ = torch.tensor(bt[:, -self.s_dim:], dtype=torch.float32)

        # update critic
        q = self.critic_eval(bs, ba)
        with torch.no_grad():
            ba_ = self.actor_target(bs_)
            q_ = self.critic_target(bs_, ba_)
            q_target = br + GAMMA * q_

        critic_loss = F.mse_loss(q, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actor_loss = -q.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(),
                                       self.actor_eval.parameters()):
            target_param.data.copy_((1 - TAU) * target_param.data +
                                    TAU * param.data)

        for target_param, param in zip(self.critic_target.parameters(),
                                       self.critic_eval.parameters()):
            target_param.data.copy_((1 - TAU) * target_param.data +
                                    TAU * param.data)

    def store_transition(self, s, a, r, s_):
        trasition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = trasition
        self.pointer += 1


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high  # 值域范围

ddpg = DDPG(s_dim, a_dim, a_bound)

var = 3
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)  # 增加随机性
        s_, r, done, info = env.step(a)

        # 值域映射
        # a = ddpg.choose_action(s)
        # a_mapping = map_function(a)
        # env.step(a_mapping)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995
            ddpg.learn()

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS - 1:
            break

print('Running time: ', time.time() - t1)
