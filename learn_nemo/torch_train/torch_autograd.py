import torch

# 自变量，一会要求导。
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 一维热传导方程
y = x ** 3 + 2 * x**2

dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]  # 求一阶导数
print("x=2.0, 一阶导数dy/dx:", dy_dx)

d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=False)[0]  # 求二阶导数
print("x=2.0, 二阶导数d²y/dx²:", d2y_dx2)

# 多元波函数, t,x同数量,则求导数时,t,x一一对应,共3个状态.
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
t = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)

u = torch.sin(x -  2* t)

du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # 求一阶导数
du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # 求一阶导数
print("x=1.0, t=0.5, 一阶导数du/dx:", du_dx)
print("x=1.0, t=0.5, 一阶导数du/dt:", du_dt)

d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=False)[0]  # 求二阶导数
print("x=1.0, t=0.5, 二阶导数d²u/dx²:", d2u_dx2)

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)

x1 = x.view(-1, 2)  # 将 x 转换为列向量
x2 = x.reshape(-1, 2)  # 将 x 转换为列向量
print("x1 shape:", x1.shape)  # 输出 x1 的形状
print("x2 shape:", x2.shape)  # 输出 x2 的形状

x = torch.tensor([[1.0, 2.0, 3.0],[4.0,5.0,6.0]], requires_grad=True)
x_t = x.T
print("x shape:", x.shape)  # 输出 x 的形状
print("x_t shape:", x_t.shape)  # 输出 x_t 的形状
x_reshape = x_t.reshape(-1, 1)
print("x_reshape shape:", x_reshape.shape)
print(x_reshape)  # 输出 x_reshape 的形状