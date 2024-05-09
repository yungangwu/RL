import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


np.random.seed(1)

class SumTree(object):
    # 建立tree和data
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # 非叶子节点有capacity-1个

        self.data = np.zeros(capacity, dtype=object)

    # 当有新的sample时，添加进tree和data
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    # 当sample被train时，有了新的TD-error，就在tree中更新
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0: # 找到父节点，然后更新其值
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # 根据选取的v点抽取样本
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1 # 节点的左右孩子
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    # 获取sum(priorities)
    @property
    def total_p(self):
        return self.tree[0]


class DQN(nn.Moudule):
    def __init__(self, n_actions, n_features):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        self.fc2 = nn.Linear(20, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNPriortizedReplay:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.005,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=500,
        memory_size=10000,
        batch_size=32,
        e_greedy_increment=None,
        output_graph=False,
        prioritized=True,
    ):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized

        self.learn_step_counter = 0

        self.eval_net = DQN(n_actions, n_features)
        self.target_net = DQN(n_actions, n_features)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=learning_rate)

        self.memory_counter = 0
        self.memory = np.zeros((memory_size, n_features * 2 + 2))
        
        self.cost_his = []
    
    def store_transition(self, s, a, r, s_):
        if self.prioritized:
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1
    
    def choose_action(self, observation):
        observation = torch.Tensor(observation[np.newaxis, :])

        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation)
            action = torch.argmax(actions_value).item()
        else:
            action = np.random.randint(0, self.n_actions)
        
        return action
    
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        s = torch.Tensor(batch_memory[:, :self.n_features])
        a = torch.LongTensor(batch_memory[:, self.n_features].astype(int))
        r = torch.Tensor(batch_memory[:, self.n_features + 1])
        s_ = torch.Tensor(batch_memory[:, -self.n_features:])

        q_eval = self.eval_net(s).gather(1, a.unsqueeze(1).long())
        q_next = self.target_net(s_).detach()
        q_target = r + self.gamma * q_next.max(1)[0]

        loss = nn.MSELoss()(q_eval, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
