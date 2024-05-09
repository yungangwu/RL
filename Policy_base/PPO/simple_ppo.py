import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip',
         epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]

A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10


class ACNet(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(ACNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound

        self.a1 = nn.Linear(s_dim, 200)
        self.mean = nn.Linear(200, 1)
        self.log_std = nn.Linear(200, 1)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)

        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a = F.relu(self.a1(x))
        mean = self.a_bound * F.tanh(self.mean(a))
        log_std = self.log_std(a)
        c = F.relu(self.c1(x))
        v = self.v(c)
        return mean, log_std, v

    def choose_action(self, s_dim):
        mean, log_std, _ = self.forward(s_dim)
        m = self.distribution(mean.view(1, ).data, log_std.view(1, ).data)
        return m.sample().numpy()

    def loss_critic(self, s_dim, reward):
        mean, log_std, v = self.forward(s_dim)
        td_error = reward - v
        c_loss = td_error.pow(2)
        return c_loss


class PPO:
    def __init__(self, s_dim, a_dim,
                 a_bound) -> None:  # critic只有一个，actor分为了old和new
        self.pi_acnet = ACNet(s_dim, a_dim, a_bound)
        self.old_acnet = ACNet(s_dim, a_dim, a_bound)
        self.optimpi = optim.Adam(self.pi_acnet.parameters())

    def update(self, s_dim, a_dim, r):
        self.update_old_ac()

        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.comput_actor_loss(s_dim, a_dim, r)
                if kl > 4 * METHOD['kl_target']:
                    break

            if kl < METHOD['kl_target'] / 1.5:
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2

            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)
        else:
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.comput_actor_loss(s_dim, a_dim, r)

        for _ in range(C_UPDATE_STEPS):
            self.comput_critic_loss(s_dim, r)

    def comput_actor_loss(self, s_dim, a_dim, r):
        s = torch.tensor(s_dim, dtype=torch.float32)
        a = torch.tensor(a_dim, dtype=torch.float32)
        r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(1)

        mean, log_std, v = self.pi_acnet(s)
        td_error = r_t - v
        pi_m = self.pi_acnet.distribution(mean, log_std)
        pi_log_prob = pi_m.log_prob(a)

        old_mean, old_log_std, _ = self.old_acnet(s)
        old_m = self.old_acnet.distribution(old_mean, old_log_std)
        old_log_prob = old_m.log_prob(a)

        ratio = pi_log_prob - old_log_prob
        surr = ratio * td_error
        surr1 = torch.clamp(ratio, 1.0 - METHOD['epsilon'],
                            1.0 + METHOD['epsilon'])

        if METHOD['name'] == 'kl_pen':
            kl = torch.distributions.kl_divergence(old_m, pi_m)
            self.kl_mean = torch.mean(kl)
            loss = -torch.mean(surr - METHOD["lam"] * kl)
        else:
            loss = -torch.mean(torch.min(surr, surr1))

        self.optimpi.zero_grad()
        loss.backward()
        self.optimpi.step()

        if METHOD['name'] == 'kl_pen':
            return loss.item(), kl.item()
        else:
            return loss.item(), None

    def comput_critic_loss(self, states, rewards):  # actor与critic都进行了反向传播，可能有
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)

        values = self.pi_acnet.loss_critic(states).squeeze()
        loss = F.mse_loss(values, rewards)

        self.optimpi.zero_grad()
        loss.backward()
        self.optimpi.step()

        return loss.item()

    def update_old_ac(self):
        self.old_acnet.load_state_dict(self.pi_acnet.state_dict())

    def get_v(self, s_dim):
        s = torch.tensor(s_dim, dtype=torch.float32)
        _, _, v = self.pi_acnet(s)
        return v

    def choose_action(self, s_dim):
        s = torch.tensor(s_dim, dtype=torch.float32)
        a = self.pi_acnet.choose_action(s)
        return np.clip(a, -2, 2)