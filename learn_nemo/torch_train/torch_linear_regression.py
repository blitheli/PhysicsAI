import torch
import torch.nn as nn

# 生成 100 个在 -1 到 1 之间均匀分布的点，并将其形状调整为 (100, 1)
x = torch.linspace(-1, 1, 100).view(-1, 1)

# 真实的线性关系为 y = 3x + 2，并添加一些噪声
y = 3 * x + 2 + torch.randn_like(x) * 0.1

model = nn.Linear(1, 1)  # 输入维度为 1，输出维度为 1
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  


for epoch in range(100):  # 训练 100 个 epoch
    optimizer.zero_grad()  # 清空梯度
    pred = model(x)  # 前向传播, 计算预测值
    loss = criterion(pred, y)  # 计算损失
    loss.backward()  # 反向传播, 计算梯度
    optimizer.step()  # 更新权重参数w,b
    if (epoch+1) % 10 == 0:
        print(f'Weights: {model.weight.item():.4f}, Bias: {model.bias.item():.4f}')
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

w = model.weight.item()  # 获取权重参数 w
b = model.bias.item()  # 获取偏置参数 b
print(f'训练完成后，模型参数 w: {w:.4f}, b: {b:.4f}')        

