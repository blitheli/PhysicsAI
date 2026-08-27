import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个物理信息神经网络（PINN）模型
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = PINN()  # 实例化模型

optimizer = optim.Adam(model.parameters(), lr=0.01)  # 定义优化器
criterion = nn.MSELoss()  # 定义损失函数为均方误差

# 训练模型,仅靠物理约束条件来训练模型
for epoch in range(1500):  # 训练 1000 个 epoch

    # 随机生成训练数据,0,1之间均匀分布的点
    x_pde = torch.rand(100, 1, requires_grad=True)  # 生成训练数据

    u_pde = model(x_pde)  # 前向传播，计算预测值

    # 1,2阶求导
    du_dx = torch.autograd.grad(u_pde, x_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_pde, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

    # 物理控制方程的残差: 强迫 二阶导数为-2
    loss_pde = criterion(d2u_dx2, -2 * torch.ones_like(d2u_dx2))

    # 训练数据的边界条件
    x_bc_left = torch.tensor([[0.0]], requires_grad=False)  # 左边界条件
    x_bc_right = torch.tensor([[1.0]], requires_grad=False)  # 右边界条件

    u_bc_left = model(x_bc_left)
    u_bc_right = model(x_bc_right)

    loss_bc = criterion(u_bc_left, torch.tensor([[0.0]])) + criterion(u_bc_right, torch.tensor([[1.0]]))

    loss = loss_pde + loss_bc * 10  # 总损失为 PDE 损失和边界条件损失的加权和

    optimizer.zero_grad()   # 清空梯度
    loss.backward()         # 反向传播，计算梯度
    optimizer.step()        # 更新权重参数

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}')

# 验证模型
x_eval= torch.tensor([[0.5]])  # 验证数据
y_eval = model(x_eval)  # 前向传播，计算预测值
y_real = -0.5**2 + 2 * 0.5  # 真实值
print(f'验证数据 x={x_eval.item():.2f}, 预测值 y={y_eval.item():.4f}, 真实值 y={y_real:.4f}')  #