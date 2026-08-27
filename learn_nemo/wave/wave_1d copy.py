import numpy as np
import torch
import torch.nn as nn
from physicsnemo.models.mlp import FullyConnected


def pde_loss(model, x, t, c, mse_loss):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    inputs = torch.cat([x, t], dim=1)
    u = model(inputs)

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_tt = torch.autograd.grad(
        u_t,
        t,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
        retain_graph=True,
    )[0]

    pde_residual = u_tt - (c**2) * u_xx
    return mse_loss(pde_residual, torch.zeros_like(pde_residual))


def Run_wave_pinn():

    # 硬件设备及高性能网络初始化
    #--------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"使用Device: {device}")

    # 实例化PhysicsNemo骨干网: 输入(x,t), 输出u
    # 采用6层隐藏层，每层128个神经元的全连接网络，激活函数为torch.sin(Siren)
    wave_net = FullyConnected(
        in_features=2,
        out_features=1,
        num_layers=6,
        layer_size=128,
        activation_fn="tanh").to(device)


    L = np.pi   # 空间范围
    T = 2.0 * np.pi  # 时间范围
    c = 1.0  # 波速

    # 数据集生成
    #--------------------------------------------------------------------------
    num_interior = 4096  # 内部点数量(均匀分布的随机数)(num_interior, 1)
    num_ic = 2048   # 初始条件点数量(均匀分布的随机数)(num_ic, 1)
    num_bc = 2048   # 边界条件点数量(均匀分布的随机数)(num_bc, 1)

    # 验证集准备(网格解析解)
    dx, dt  = 0.02, 0.02
    x_val = np.arange(0, L, dx)
    t_val = np.arange(0, T, dt)
    # 生成网格数据，X_mesh和T_mesh分别表示空间和时间的网格坐标
    X_mesh, T_mesh = np.meshgrid(x_val, t_val)
    # 将网格数据展平为一维数组，以便后续处理
    X_flat = np.expand_dims(X_mesh.flatten(), axis=-1)
    T_flat = np.expand_dims(T_mesh.flatten(), axis=-1)

    u_true = np.sin(X_flat) * (np.cos(T_flat) + np.sin(T_flat))  # 解析解

    # 验证集输入和真实值,这里是把x,t拼接成一个二维数组,每一行是一个点的(x,t)坐标
    val_input = torch.tensor(np.concatenate([X_flat, T_flat], axis=1), dtype=torch.float32, device=device)
    val_true= torch.tensor(u_true, dtype=torch.float32, device=device)

    # 优化器选择与训练循环
    optimizer = torch.optim.Adam(wave_net.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    print("开始训练...")
    for epoch in range(3000):
        optimizer.zero_grad()

        x_int = torch.rand(num_interior, 1, dtype=torch.float32, requires_grad=True, device=device) * L
        t_int = torch.rand(num_interior, 1, dtype=torch.float32, device=device) * T
        loss_pde = pde_loss(wave_net, x_int, t_int, c, mse_loss)

        # 初始条件: u(x,0) = sin(x), ut(x,0) = sin(x)
        x_ic = torch.rand(num_ic, 1, dtype=torch.float32, requires_grad=True, device=device) * L
        t_ic = torch.zeros(num_ic, 1, dtype=torch.float32, requires_grad=True, device=device)
        u_ic_target = torch.sin(x_ic).detach()
        ut_ic_target = torch.sin(x_ic).detach()

        input_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic_pred = wave_net(input_ic)
        loss_ic = mse_loss(u_ic_pred, u_ic_target)

        # 计算初始条件的时间导数
        u_ic_pred_t = torch.autograd.grad(
            u_ic_pred,
            t_ic,
            grad_outputs=torch.ones_like(u_ic_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        loss_ic_ut = mse_loss(u_ic_pred_t, ut_ic_target)

        # 边界条件: u(0,t) = 0, u(L,t) = 0
        t_bc = torch.rand(num_bc, 1, dtype=torch.float32, device=device) * T
        x_bc_left = torch.zeros(num_bc, 1, dtype=torch.float32, device=device)
        x_bc_right = L * torch.ones(num_bc, 1, dtype=torch.float32, device=device)

        input_bc_left = torch.cat([x_bc_left, t_bc], dim=1)
        input_bc_right = torch.cat([x_bc_right, t_bc], dim=1)
        u_bc_l_pred = wave_net(input_bc_left)
        u_bc_r_pred = wave_net(input_bc_right)
        loss_bc = mse_loss(u_bc_l_pred, torch.zeros_like(u_bc_l_pred)) + mse_loss(u_bc_r_pred, torch.zeros_like(u_bc_r_pred))
        # 总损失函数
        total_loss = loss_pde + loss_ic + loss_ic_ut + loss_bc
        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/1000], Total Loss: {total_loss.item():.6f}')

            # 验证模型在网格点上的表现
            wave_net.eval()
            with torch.no_grad():
                u_val_pred = wave_net(val_input)
                val_loss = mse_loss(u_val_pred, val_true)
                print(f'Validation Loss: {val_loss.item():.6f}')
            wave_net.train()
# 数据
if __name__ == "__main__":
    Run_wave_pinn()
   