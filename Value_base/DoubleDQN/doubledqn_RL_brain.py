import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLAYCE_ITET = 100
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 杆子能做的动作
N_STATES = env.observation_space.shape[0]   # 杆子能获取的环境信息数

class Net(nn.Module):
    def __init__(self, n_actions, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc1.weight.data.normal_(0, 0.1) # weight initialization
        self.out = nn.Linear(10, n_actions)
        self.out.weight.data.normal_(0, 0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DoubleDQN:
    def __init__(self, n_actions,
                       n_features,
                       learning_rate=0.01, 
                       reward_decay=0.9, 
                       e_greedy=0.9, 
                       replace_target_iter=200, 
                       memory_size=2000,
                       batch_size=64,
                       double_q=True,
                ):
        self.n_actions = n_actions
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.eval_net = Net(n_actions, n_features)
        self.target_net = Net(n_actions, n_features)

        self.replace_target_iter = replace_target_iter
        self.memory_counter = 0
        self.memory = np.zeros((memory_size, n_features * 2 + 2)) # memory中要存两obs，一个action和一个reward，所以尺寸是obs*2+2
        self.optimizer = torch.optim.Adam(self.eval_net.paramters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.cost_his = []

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0) # 用于在指定维度上插入一个大小为1的维度
        action_values = self.eval_net.forward(x)
        action = torch.max(action_values, 1)[1].data.numpy()[0, 0]
        
        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0
        
        self.running_q = self.running_q*0.99 + 0.01*np.max(actions_value)
        self.q.append(self.running_q)
        
        if np.random.uniform() > EPSILON:
            action = np.random.randint(0, self.n_actions)
        return action
    
    def store_transiton(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) # 用于水平（按列）拼接多个数组
        # 如果记忆库满了就覆盖老数据
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.learn_step_counter += 1

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_features])
        b_a = torch.LongTensor(b_memory[:, self.n_features:self.n_features+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_features+1:self.n_features+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_features:])
        q_next, q_eval4next = self.target_net(b_s).detach(), self.eval_net(b_s).detach()
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)
        q_target = b_r + self.gamma * selected_q_next
        # q_eval = self.eval_net(b_s).gather(1, b_a)
        # q_target_next = self.target_net(b_s_).detach()
        # q_target = b_r + self.gamma * q_target_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)
        self.cost_his.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

