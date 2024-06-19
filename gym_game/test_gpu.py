import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 创建模型实例
input_size = 784  # 例如，对于28x28的图像输入
hidden_size = 128
output_size = 10  # 例如，对于10类分类任务
model = SimpleNN(input_size, hidden_size, output_size)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 将模型移动到GPU上
model.to(device)

# 创建一些随机输入数据并移动到GPU上
batch_size = 64
input_data = torch.randn(batch_size, input_size).to(device)
target_data = torch.randint(0, output_size, (batch_size,)).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 进行前向传播计算
output = model(input_data)

# 计算损失
loss = criterion(output, target_data)

# 打印输出和损失
print("Output:", output)
print("Loss:", loss.item())

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Training step completed.")
