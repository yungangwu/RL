import torch
import torch.nn as nn

# net部分将obs等信息转换为特征向量，然后通过全连接层进行分类
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