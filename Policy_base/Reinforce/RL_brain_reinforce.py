import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Moudle):
    def __init__(self, n_actions, n_features):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(n_features, 10)
        self.fc1.weight.data.normal_(0, 0.3) # weight initialization
        self.fc2 = nn.Linear(10, n_actions)
        self.fc2.weight.data.normal_(0, 0.3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        return F.softmax(x, dim=1)

class PolicyGradient:

    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.policy_net = Network(self.n_actions, self.n_features)

        self.optimizer = torch.optim.Adam(self.policy_net.paramters(), lr=learning_rate)
        self.loss_func = F.cross_entorpy()

    def choose_action(self, observation):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        prob_weights = self.policy_net(observation_tensor.unsqueenze(0))

        prob_weights = prob_weights.detach().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self, s, a, r, s_):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        obs_tensor = torch.tensor(self.ep_obs, dtype=torch.float32)
        acts_tensor = torch.tensor(self.ep_as, dtype=torch.int64)
        vt_tensor = torch.tensor(discounted_ep_rs_norm, dtype=torch.float32)

        self.optimizer.zerp_grad()
        log_probs = self.model(obs_tensor)
        selected_log_probs = log_probs[torch.arange(len(acts_tensor)), acts_tensor]
        loss = -(selected_log_probs * vt_tensor).mean()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        return discounted_ep_rs
