import torch
import torch.nn as nn
import torch.nn.functional as F

'''
你的当前代码是“数据生成器”，它只是为了演示 FNO 学习热扩散映射。
它并不是完整的 PDE 求解器，所以：
    - 没有严格考虑边界
    - 没有热源项
    - 没有真实物理边界类型

'''

# 随机生成一个批次的初始温度场 u0，并计算经过一步热扩散后的温度场 u1,范围:[-1,1]
# 作为训练数据，供 FNO 模型学习热扩散规律
def make_heat_step_batch(batch_size=32, size=16, alpha=0.1, dt=0.01):
    """Generate a batch of initial temperature fields and one-step heat evolution."""
    # 随机生成初始温度场 u0，范围在 [-1, 1]
    # 这不是物理上的真实温度单位，而是“归一化后的温度场”。之后模型学习它随时间扩散的规律
    u0 = torch.rand(batch_size, 1, size, size) * 2.0 - 1.0
    # 离散化的二维拉普拉斯算子，用于计算 u0 的空间二阶导数，近似热方程中的扩散项
    lap = F.conv2d(
        u0,
        torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]),
        padding=1,
    )
    # 热方程一步更新
    u1 = u0 + alpha * dt * lap
    return u0, u1

# FNO 模块中的傅里叶块，负责在频域进行低频滤波和线性变换
class FourierBlock(nn.Module):
    """A tiny FNO-style spectral block using low-frequency filtering."""

    def __init__(self, channels, modes=4):
        super().__init__()
        # 可学习的低频模式数量
        self.modes = modes

        # 可学习参数矩阵
        self.filter = nn.Parameter(torch.randn(channels, modes, modes) * 0.1)

    # 前向传播函数
    # 先做傅里叶变换, 在频域做线性变换, 再逆变换回空间场
    def forward(self, x):
        b, c, h, w = x.shape
        # 对输入张量进行二维快速傅里叶变换，得到频域表示，因为实数傅里叶变换只保留一半的频率信息。
        # 返回的是[batch, channels, H, W//2+1]
        x_fft = torch.fft.rfft2(x, s=(h, w))
        m1 = min(self.modes, x_fft.shape[-2])
        m2 = min(self.modes, x_fft.shape[-1])

        # 只保留低频部分，并与可学习的滤波器进行逐元素相乘，实现低频滤波和线性变换
        x_low = x_fft[:, :, :m1, :m2]
        # 这里将学习到的滤波器和当前输入的通道数对齐。
        filt = self.filter[:c, :m1, :m2].view(1, c, m1, m2)
        x_out_fft = torch.zeros_like(x_fft)
        x_out_fft[:, :, :m1, :m2] = x_low * filt

        # 对频域表示进行二维逆快速傅里叶变换，得到时域输出
        x_out = torch.fft.irfft2(x_out_fft, s=(h, w))
        return x_out

# 定义一个小型的 FNO 模型，包含升维卷积、傅里叶块和降维卷积
class TinyFNO(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=16, modes=4):
        super().__init__()
        # 升维卷积，将输入的单通道温度场升维到隐藏通道数
        self.lift = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        # 这里把升维后的特征场送进傅里叶块。
        self.fourier = FourierBlock(hidden_channels, modes=modes)
        # 降维卷积，将隐藏通道数的特征场降维回输出通道数
        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    # 输入场 -> 升维 -> 傅里叶全局学习 -> 残差 -> 激活 -> 投影 -> 输出场
    def forward(self, x):
        x = self.lift(x)
        # 典型的 residual 连接。先保留原始特征 + 再加上傅里叶处理后的结果
        x = x + self.fourier(x)
        # 这是非线性激活
        x = torch.relu(x)
        # 输出投影，输入：16个通道,1个输出通道
        x = self.proj(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建 FNO 模型并将其移动到 GPU（如果可用）
    model = TinyFNO().to(device)
    # 定义优化器，使用 Adam 优化器，学习率为 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(200):

        # 随机生成一个批次的初始温度场 u0，并计算经过一步热扩散后的温度场 u1
        # 作为训练数据，供 FNO 模型学习热扩散规律
        u0, u1 = make_heat_step_batch(batch_size=16, size=16, alpha=0.1, dt=0.01)
        u0, u1 = u0.to(device), u1.to(device)

        # 前向传播，计算预测值和损失,这里shape:16,1,16,16
        pred = model(u0)
        loss = F.mse_loss(pred, u1) # 对所有的batch,所有通道,所有空间位置一起算平均误差

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 49:
            print(f"epoch={epoch:02d} loss={loss.item():.6f}")

    # 测试模型在新数据上的表现
    with torch.no_grad():
        test_u0, test_u1 = make_heat_step_batch(batch_size=4, size=16, alpha=0.1, dt=0.01)
        test_u0 = test_u0.to(device)
        pred = model(test_u0)
        print("example prediction shape:", tuple(pred.shape))
        print("example mse:", F.mse_loss(pred, test_u1.to(device)).item())
