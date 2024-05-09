import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from env import GomokuEnv

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()

        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height) # 输出的动作概率是以整个棋盘为动作空间的
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * board_width * board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * board_width * board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val)) # tanh的值域是-1，1

        return x_act, x_val # 当前棋面选择的动作概率，当前棋面的价值

class PolicyValueNet:
    def __init__(self, model_file=None) -> None:

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.l2_const = 1e-4

        self.policy_value_net = Net().to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, board: GomokuEnv):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape( # np.ascontiguousarray()是一个将内存不连续存储的数组转换为内存连续存储的数组
                -1, 4, board_width, board_height))
        log_act_probs, value = self.policy_value_net(
            Variable(torch.from_numpy(current_state)).to(self.device).float()
        )
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = Variable(torch.FloatTensor(state_batch).to(self.device)) # 统一用Variable包起来，并放到相同的硬件设备上
        mcts_probs = Variable(torch.FloatTensor(mcts_probs).to(self.device))
        winner_batch = Variable(torch.FloatTensor(winner_batch).to(self.device))

        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr # 训练过程中修改学习率

        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch) # winner_batch可以看作是奖励，value则是当前棋面的价值，采用mse的方式计算损失，目的是让策略价值网络输出的状态价值更接近真实的胜者
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1)) # 计算mcts搜索找到的动作概率与网络输出的动作概率之间的差距，目的是让策略价值网络输出的动作概率更接近mcts算法得到的动作概率
                                                                            # AlphaZero不是直接用mcts找到一个最佳的动作吗？这里看起来是采用mcts来指导网络输出
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

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