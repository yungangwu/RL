import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import threading, gym
import matplotlib.pyplot as plt
from torch.multiprocessing import Queue, Event

EPSILON = 0.2
EP_MAX = 1000
EP_LEN = 500
UPDATE_STEP = 15
GAME = 'CartPole-v0'
MIN_BATCH_SIZE = 64
GAMMA = 0.9
N_WORKER = 4
SHOULD_STOP = False


class Actor(nn.Module):
    def __init__(self, s_dim, a_bound) -> None:
        super(Actor, self).__init__()
        self.a_bound = a_bound

        self.f1 = nn.Linear(s_dim, 100)
        self.mean = nn.Linear(100, 1)
        self.log_std = nn.Linear(100, 1)

        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a = F.relu(self.f1(x))
        mean = self.a_bound * F.tanh(self.mean(a))
        log_std = self.log_std(a)

        return mean, log_std


class Critic:
    def __init__(self, s_dim) -> None:
        super(Critic, self).__init__()
        self.f1 = nn.Linear(s_dim, 100)
        self.f2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = self.f2(x)
        return x


class PPO:
    def __init__(self, s_dim, a_bound) -> None:
        self.s_dim = s_dim

        self.pi_actor = Actor(s_dim, a_bound)
        self.old_actor = Actor(s_dim, a_bound)
        self.critic = Critic(s_dim)

        self.optim_actor = optim.Adam(self.pi_actor.parameters())
        self.optim_critic = optim.Adam(self.old_actor.parameters())

    def comput_actor_loss(self, s, a, r):
        v = self.critic(s)
        td_error = r - v

        pi_mean, pi_log_std = self.pi_actor(s)
        pi_distribution = self.pi_actor.distribution(pi_mean, pi_log_std)
        pi_log_prob = pi_distribution.log_prob(a)

        old_mean, old_log_std = self.old_actor(s)
        old_distribution = self.old_actor.distribution(old_mean, old_log_std)
        old_log_prob = old_distribution.log_prob(a)

        ratio = pi_log_prob / (old_log_prob + 1e-5)
        surr = ratio * td_error

        temp = torch.clip(ratio, 1. - EPSILON, 1. + EPSILON) * td_error
        a_loss = -torch.mean(torch.min(temp, surr))

        self.optim_actor.zero_grad()
        a_loss.backward()
        self.optim_actor.step()

    def comput_critic_loss(self, s, r):
        v = self.critic(s)
        td_error = r - v
        c_loss = torch.mean(torch.square(td_error))

        self.optim_critic.zero_grad()
        c_loss.backward()
        self.optim_critic.step()

    def update_old_actor(self):
        self.old_actor.load_state_dict(self.pi_actor.state_dict())

    def update(self, s, a, r):
        global GLOBAL_UPDATE_COUNTER
        while not SHOULD_STOP:
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()
                self.update_old_actor()
                data = [QUEUE.get() for _ in range(QUEUE.qszie())]
                data = np.vstack(data)
                s, a, r = data[:, :self.s_dim], data[:, self.s_dim:self.s_dim +
                                                     1].ravel(), data[:, -1]
                for _ in range(UPDATE_STEP):
                    self.comput_actor_loss(s, a, r)
                    self.comput_critic_loss(s, r)
                UPDATE_EVENT.clear()
                GLOBAL_UPDATE_COUNTER = 0
                ROLLING_EVENT.set()

    def choose_action(self, s):
        pi_mean, pi_log_std = self.pi_actor(s)
        pi_distribution = self.pi_actor.distribution(pi_mean, pi_log_std)
        action = pi_distribution.sample().numpy()
        return action

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.critic(s)[0, 0]


class Worker:
    def __init__(self, wid) -> None:
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not SHOULD_STOP:
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():
                    ROLLING_EVENT.wait()
                    buffer_s, buffer_a, buffer_r = [], [], []
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                if done: r = -10
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.ppo.get_v(s)

                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)

                    discounted_r.reverse()
                    bs, ba, br = np.vstack(buffer_s), np.vstack(
                        buffer_a), np.vstack(buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()
                        UPDATE_EVENT.set()

                    if GLOBAL_EP >= EP_MAX:
                        COORD.request_stop()
                        break

                    if done: break

            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 +
                                        ep_r * 0.1)
            GLOBAL_EP += 1


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    workers = [Worker(wid=i) for i in range(4)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    SHOULD_STOP = True
    QUEUE = Queue()
    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)

    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    for t in threads:
        t.join()

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.show()
    env = gym.make('CartPole-v0')
    while True:
        s = env.reset()
        for t in range(1000):
            env.render()
            s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
            if done:
                break