import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import gym, math
from utils import set_model_init, push_and_pull, record

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200

env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]


class ActorNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(s_dim, 200)
        self.mean = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        set_model_init([self.fc1, self.mean, self.sigma])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        fc1 = F.relu6(self.fc1(x))
        mean = 2 * F.tanh(
            self.mean(fc1))  # 2 * 表示要将输出范围映射到-2，2之间，原始的tanh的输出在-1，1之间
        sigma = F.softplus(self.sigma(fc1)) + 0.001
        return mean, sigma


class CriticNet(nn.Module):
    def __init__(self, s_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Actor:
    def __init__(self, s_dim, a_dim) -> None:
        self.actor_net = ActorNet(s_dim, a_dim)

    def loss_func(self, s, a, td_error):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(a, dtype=torch.float32)
        td_error = torch.tensor(td_error, dtype=torch.float32)

        mean, sigma = self.actor_net(s)
        m = self.actor_net.distribution(
            mean.view(1, ).data,
            sigma.view(1, ).data)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(
            m.scale)  # exploration
        exp_v = log_prob * td_error.detach() + 0.005 * entropy
        actor_loss = -exp_v
        return actor_loss

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        mean, sigma = self.actor_net(s)
        m = self.actor_net.distribution(
            mean.view(1, ).data,
            sigma.view(1, ).data)
        return m.sample().numpy()


class Critic:
    def __init__(self, s_dim) -> None:
        self.critic_net = CriticNet(s_dim)

    def loss_func(self, s, r, s_):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        s_ = torch.tensor(s_, dtype=torch.float32).unsqueeze(0)
        r = torch.tensor(r, dtype=torch.float32)

        v = self.critic_net(s)
        v_ = self.critic_net(s_)

        td_error = r + GAMMA * v_ - v
        return td_error


class ACNet:
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.actor = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim)
        self.actor_optimizer = optim.Adam(self.actor.actor_net.parameters(),
                                          lr=0.01)
        self.critic_optimizer = optim.Adam(self.critic.critic_net.parameters(),
                                           lr=0.01)

    def choose_action(self, s):
        return self.actor.choose_action(s)

    def loss_func(self, s, a, r, s_):
        td_error = self.critic.loss_func(s, r, s_)
        c_loss = td_error.pow(2)

        a_loss = self.actor.loss_func(s, a, td_error)
        total_loss = (a_loss + c_loss).mean()

        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet = gnet
        self.lnet = ACNet(N_S, N_A)  # local net
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.step()
            buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                if self.name == 'w0':
                    self.env.render()

                a = self.lnet.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8.1) / 8.1)
                buffer_s_.append(s_)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.gnet, self.lnet, buffer_s, buffer_a,
                                  buffer_r, buffer_s_)
                    buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []

                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue,
                               self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = ACNet(N_S, N_A)
    gnet.actor.actor_net.share_memory()
    gnet.critic.critic_net.share_memory()

    global_ep, global_ep_r, res_queue = mp.Value('i',
                                                 0), mp.Value('d',
                                                              0.), mp.Queue()
    workers = [
        Worker(gnet, global_ep, global_ep_r, res_queue, i)
        for i in range(mp.cpu_count())
    ]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
