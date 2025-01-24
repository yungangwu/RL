import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_features, n_actions=2578):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 3000)
        self.fc1.weight.data.normal_(0, 0.1) # weight initialization
        self.out = nn.Linear(3000, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value

class ACNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        common_feature = self.feature_extractor(obs)
        action_prob = self.actor(common_feature)
        value = self.critic(common_feature)
        return action_prob, value