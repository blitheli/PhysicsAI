import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        # 创建100个在-1到1之间均匀分布的点，并将其形状调整为(100, 1)
        self.x = torch.linspace(-1, 1, 100).view(-1, 1)
        # 真实的线性关系为y = 3x + 2，并添加一些噪声
        self.y = 3 * self.x + 2 + torch.randn_like(self.x) * 0.1
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

dataset = MyDataset()
# shuffle=True表示在每个epoch开始时打乱数据，batch_size=16表示每个批次包含16个样本
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = nn.Linear(1, 1)  # 输入维度为1，输出维度为1
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器，学习率为0.1

for epoch in range(20):  # 训练20个epoch
    epoch_loss = 0.0
    num_batches = 0

    for xb, yb in dataloader:
        optimizer.zero_grad()  # 清空梯度
        pred = model(xb)  # 前向传播，计算预测值
        loss = criterion(pred, yb)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新权重参数w,b

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/20], Loss: {avg_loss:.4f}')

print(f'训练完成后，模型参数 w: {model.weight.item():.4f}, b: {model.bias.item():.4f}')