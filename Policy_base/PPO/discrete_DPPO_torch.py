import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import threading
import queue


EP_MAX = 1000
EP_LEN = 200
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
GAME = 'Pendulum-v1'
S_DIM, A_DIM = 3, 1         # state and action dimension
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(S_DIM, 100)
        self.v = nn.Linear(100, 1)
        self.mu = nn.Linear(100, A_DIM)
        self.sigma = nn.Linear(100, A_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.v(x)
        mu = 2 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        return mu, sigma, v

    def choose_action(self, s):
        # mu, sigma, _ = self.forward(torch.Tensor(s).to(DEVICE))
        with torch.no_grad():
            mu, sigma, _ = self.forward(torch.Tensor(s).to(DEVICE))
        norm_dist = torch.distributions.Normal(mu, sigma)
        a = norm_dist.sample()
        return torch.clamp(a, -2.0, 2.0).item()

    def get_v(self, s):
        _, _, v = self.forward(torch.Tensor(s).to(DEVICE))
        return v.item()

class PPOWrapper:
    def __init__(self) -> None:
        # self.device = DEVICE
        # print(f'device: {self.device}')
        # self.pi = PPO().to(self.device)
        # self.oldpi = PPO().to(self.device)

        # self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=A_LR)
        # print('ppowrapper init')

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print("Initializing PPOWrapper...")
        # self.device = DEVICE
        print(f"Device set to: {self.device}")

        # Test PPO model
        print("Initializing PPO pi...")
        self.pi = PPO()
        print("PPO pi initialized.")
        self.pi = self.pi.to('cpu')
        print("PPO pi moved to device.")

        print("Initializing PPO oldpi...")
        self.oldpi = PPO()
        print("PPO oldpi initialized.")
        self.oldpi = self.oldpi.to('cpu')
        print("PPO oldpi moved to device.")

        print("Initializing optimizer...")
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=A_LR)
        print("Optimizer initialized.")

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while True:
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                                     # wait until get batch of data
                self.update_oldpi()                                     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s = torch.tensor(data[:, :S_DIM], dtype=torch.float).to(self.device)
                a = torch.tensor(data[:, S_DIM: S_DIM + A_DIM], dtype=torch.float).to(self.device)
                r = torch.tensor(data[:, -1:], dtype=torch.float).to(self.device)

                for _ in range(UPDATE_STEP):
                    mu, std, v = self.pi(s)
                    old_mu, old_std, _ = self.oldpi(s)
                    # print('mu:', mu, 'std:', std, 's:', s)
                    dist = torch.distributions.Normal(mu, std)
                    old_dist = torch.distributions.Normal(old_mu, old_std)
                    ratio = dist.log_prob(a) / (old_dist.log_prob(a) + 1e-5)
                    adv = r - v
                    surr = ratio * adv
                    actor_loss = -torch.mean(torch.min(
                        surr,
                        torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * adv
                    ))

                    critic_loss = F.mse_loss(v, r)

                    loss = actor_loss + critic_loss
                    self.optimizer_pi.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)  # Gradient clipping
                    self.optimizer_pi.step()

                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def update_oldpi(self,):
        self.oldpi.load_state_dict(self.pi.state_dict())

class Worker:
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME)
        self.ppo_wrapper = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        print('start work...')
        while GLOBAL_EP < EP_MAX:
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo_wrapper.pi.choose_action(s)
                s_, r, done, _ = self.env.step([a])
                buffer_s.append(s)
                buffer_a.append([a])
                buffer_r.append((r + 8) / 8)                    # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo_wrapper.pi.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    print('discounted_r', discounted_r)
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r)


if __name__ == '__main__':
    print(11111111111111)
    GLOBAL_PPO = PPOWrapper()
    print(333333333)
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    print(222222222222)

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()

    for t in threads:
        t.join()

    print('threads start')
    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.show()
