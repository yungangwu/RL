import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from config.config import *

class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()

        # 添加卷积层1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 添加池化层1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 添加卷积层2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 添加池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 添加全连接层1
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        # 添加全连接层2
        self.fc2 = nn.Linear(128, 64)
        # 添加输出层
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        # 卷积层1 + 激活函数1 + 池化层1
        x = self.pool1(torch.relu(self.conv1(x)))
        # 卷积层2 + 激活函数2 + 池化层2
        x = self.pool2(torch.relu(self.conv2(x)))
        # 将卷积结果展开成向量形式
        x = x.view(-1, 64 * 3 * 3)
        # 全连接层1 + 激活函数3
        x = torch.relu(self.fc1(x))
        # 全连接层2 + 激活函数4
        x = torch.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        # 进行概率归一化操作
        return torch.softmax(x, dim=1)


class Net(nn.Module):

    def __init__(self, board_width, board_height) -> None:
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1) # 初始状态看有几层，黑子，白子，空子，当前下子？
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)

        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:
    '''policy-value network'''

    def __init__(self, model_file=None) -> None:

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.l2_const = 1e-4

        self.policy_value_net = Net(board_width=board_width, board_height=board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        '''
        input: a batch of states
        output: a batch of action probabilities and state values
        '''
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, board):
        '''
        input: board
        output: a list of (action, probability) tuple for each available action and the score of the board state
        '''
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, board_width, board_height
        ))
        log_act_probs, value = self.policy_value_net(
            torch.from_numpy(current_state).to(self.device).float()
        )
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        '''perform a training step'''
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        winner_batch = torch.FloatTensor(winner_batch).to(self.device)

        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )

        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)