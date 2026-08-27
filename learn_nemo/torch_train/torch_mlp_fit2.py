import torch
import torch.nn as nn
import torch.optim as optim

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
# 读取已训练好的模型
model.load_state_dict(torch.load('physics_fitting_model.pth', map_location=torch.device('cpu')))  # 加载模型参数

criterion = nn.MSELoss()  # 定义损失函数为均方误差
optimizer = optim.Adam(model.parameters(), lr=0.02)  # 定义优化器,自适应

model.eval()  # 设置模型为评估模式

with torch.no_grad():  # 在验证阶段不需要计算梯度
    # 生成测试数据
    x_test = torch.tensor([[0.3]])
    y_test = torch.sin(x_test) * torch.exp(-0.5*x_test)  # 测试数据对应的真实值

    test_predictions = model(x_test)  # 前向传播，计算测试集的预测值
    test_loss = criterion(test_predictions, y_test)  # 计算测试集的损失
    print(f'x输入: {x_test.item():.4f}, 预测值: {test_predictions.item():.4f}, 真实值: {y_test.item():.4f}')
    print(f'Test Loss: {test_loss.item():.4f}')