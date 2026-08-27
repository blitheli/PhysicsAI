import torch
import torch.nn as nn
import torch.optim as optim

# 求解 y = sin(x) * exp(-0.5*x) 的拟合问题

# 随机数生成训练数据, -2.0 到 4.0 之间的随机数
x_train = torch.rand(500, 1) * 6 - 2.0 #生成 500 个随机数作为训练数据
print("x_train shape:", x_train.shape)  # 输出 x_train 的形状
print(x_train.size())  # 输出 x_train 的大小

y_train = torch.sin(x_train) * torch.exp(-0.5*x_train) + torch.randn(x_train.size()) * 0.02  # 生成对应的 y_train 数据，并添加噪声

# 验证模型拟合效果（验证集）
x_val = torch.linspace(-2.0, 4.0, 100).view(-1, 1)  # 验证数据
y_val = torch.sin(x_val) * torch.exp(-0.5*x_val)

# 定义一个多层感知机（MLP）模型
class PhysicsFittingNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 使用 nn.Sequential 定义一个多层感知机（MLP）网络
        self.network = nn.Sequential(
            nn.Linear(1, 64),  # 输入层到隐藏层
            nn.Tanh(),          # 激活函数
            nn.Linear(64, 64),  # 隐藏层到隐藏层
            nn.Tanh(),          # 激活函数
            nn.Linear(64, 1)    # 隐藏层到输出层
        )

    def forward(self, x):
        return self.network(x)

model = PhysicsFittingNet()  # 实例化模型    

criterion = nn.MSELoss()  # 定义损失函数为均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 定义优化器,自适应

num_epochs = 1000 # 训练轮数

for epoch in range(num_epochs):
    # 训练
    model.train()  # 设置模型为训练模式
    predictions = model(x_train)  # 前向传播，计算预测值
    loss = criterion(predictions, y_train)  # 计算损失

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新权重参数

    # 验证
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在验证阶段不需要计算梯度
        val_predictions = model(x_val)  # 前向传播，计算验证集的预测值
        val_loss = criterion(val_predictions, y_val)  # 计算验证集的损失

    if (epoch + 1) % 100 == 0:  # 每 100 个 epoch 输出一次训练和验证损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

torch.save(model.state_dict(), 'physics_fitting_model.pth')  # 保存模型参数
print("模型参数已保存到 physics_fitting_model.pth")