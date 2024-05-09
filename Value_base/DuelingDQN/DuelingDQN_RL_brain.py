import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from DQN.DQN_RL_brain import DeepQNetwork

class DuelingNetwork(nn.Module):
    def __init__(self, n_actions, n_features, dueling=True):
        super(DuelingDQN, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.dueling = dueling

        self.fc1 = nn.Linear(n_features, 20)

        if dueling:
            self.fc_value = nn.Linear(20, 1)
            self.fc_advantage = nn.Linear(20, n_actions)
        else:
            self.fc2 = nn.Linear(20, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))

        if self.dueling:
            value = self.fc_value(x)
            advantage = self.fc_advantage(x)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q = self.fc2(x)
        
        return q

class DuelingDQN(DeepQNetwork):
    def __init__(self, dueling=True):
        super(DuelingDQN, self).__init__()

        self.eval_net = DuelingNetwork(self.n_actions, self.n_features)
        self.target_net = DuelingNetwork(self.n_actions, self.n_features)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.learning_rate)


