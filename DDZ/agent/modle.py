import torch
import torch.nn as nn
import torch.nn.functional as F

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