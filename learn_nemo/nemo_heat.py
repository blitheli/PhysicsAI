import torch
import torch.nn as nn
import torch.optim as optim
from sympy import Symbol, Function
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.eq.pde import PDE

# 使用原生的nn.Module来定义模型, 3层
class PhysicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.mlp(x)

model = PhysicsNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义 1D 热传导 PDE: d2u/dx2 + 2.0 = 0
class HeatEquation1D(PDE):
    def __init__(self):
        super().__init__()
        self.dim = 1
        x_sym = Symbol('x')
        u_sym = Function('u')(x_sym)
        self.equations = {
            'heat_equation': u_sym.diff(x_sym, 2) + 2.0,
        }

# 使用当前版本 PhysicsInformer 的参数接口
informer = PhysicsInformer(
    required_outputs=['heat_equation'], # 指定需要计算的 PDE 残差项
    equations=HeatEquation1D(),
    grad_method='autodiff',
)

# 训练模型,仅靠物理约束条件来训练模型
mse_criterion = nn.MSELoss()
for epoch in range(1500):
    optimizer.zero_grad()   # 梯度清空

    # 随机在0到1之间生成训练数据
    x_raw = torch.rand(128, 1)
    x_input = x_raw.clone().detach().requires_grad_(True)  # 需要梯度
    u_pred = model(x_input)

    # autodiff 模式需要 'coordinates' 和模型输出变量 'u'
    output_dict = informer.forward({'coordinates': x_input, 'u': u_pred})

    # 计算PDE损失, 方程残差项
    loss_pde = mse_criterion(output_dict['heat_equation'], torch.zeros_like(output_dict['heat_equation']))

    # 训练数据的边界条件
    x_left = torch.tensor([[0.0]], requires_grad=True)
    u_left_pred = model(x_left)
    loss_bc_left = mse_criterion(u_left_pred, torch.tensor([[0.0]]))
    x_right = torch.tensor([[1.0]], requires_grad=True)
    u_right_pred = model(x_right)
    loss_bc_right = mse_criterion(u_right_pred, torch.tensor([[1.0]]))

    # 总损失为 PDE 损失和边界条件损失的加权和
    loss_bc = loss_bc_left + loss_bc_right
    total_loss = loss_pde + 10 * loss_bc
    # 反向传播和优化
    total_loss.backward()
    optimizer.step()

    if(epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1500], Total Loss: {total_loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}')

# 验证模型
model.eval()

x_eval = torch.tensor([[0.5]], requires_grad=True)
y_eval = model(x_eval).item()
y_real = 0.75

print(f'Predicted value at x=0.5: {y_eval:.6f}, Real value: {y_real:.6f}')