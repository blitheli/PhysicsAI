import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


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

total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% 用于
eval_size = total_size - train_size  # 20% 用于验证

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# shuffle=True表示在每个epoch开始时打乱数据，batch_size=16表示每个批次包含16个样本
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

model = nn.Linear(1, 1)  # 输入维度为1，输出维度为1
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器，学习率为0.1

save_path = "/workspace/linear_model.pth"

for epoch in range(20):  # 训练20个epoch

    train_loss = 0.0
    num_batches = 0

    model.train()  # 设置模型为训练模式

    for xb, yb in train_dataloader:
        optimizer.zero_grad()  # 清空梯度
        pred = model(xb)  # 前向传播，计算预测值
        loss = criterion(pred, yb)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新权重参数w,b

        train_loss += loss.item()
        num_batches += 1

    avg_train_loss = train_loss / num_batches

    model.eval()  # 设置模型为评估模式
    eval_loss = 0.0
    num_eval_batches = 0
    with torch.no_grad():  # 在评估模式下不需要计算梯度
        for xb, yb in eval_dataloader:
            pred = model(xb)  # 前向传播，计算预测值
            loss = criterion(pred, yb)  # 计算损失
            eval_loss += loss.item()
            num_eval_batches += 1
    avg_eval_loss = eval_loss / num_eval_batches

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}')

print(f'训练完成后，模型参数 w: {model.weight.item():.4f}, b: {model.bias.item():.4f}')

# 保存模型参数
torch.save({
    'epoch': 20,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'eval_loss': avg_eval_loss,
}, save_path)

print(f'模型参数已保存到: {save_path}')

# 加载模型参数
loaded_model = nn.Linear(1, 1)
checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

with torch.no_grad():
    sample = torch.tensor([[0.5], [1.0]])
    preds = loaded_model(sample)
    print('加载后的预测结果:', preds)

print('模型加载成功。')