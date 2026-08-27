import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def save_solution_plots(epoch, model, X_mesh, x_val, t_val, val_input, val_true, output_dir):
    model.eval()
    with torch.no_grad():
        u_val_pred = model(val_input).squeeze(-1).cpu().numpy()
        u_true_np = val_true.squeeze(-1).cpu().numpy()

    pred_grid = u_val_pred.reshape(X_mesh.shape)
    true_grid = u_true_np.reshape(X_mesh.shape)
    err_grid = pred_grid - true_grid

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    extent = [0.0, np.pi, 0.0, 2.0 * np.pi]
    cmap = "viridis"

    im0 = axes[0, 0].imshow(pred_grid, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    axes[0, 0].set_title("Prediction surface")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.9)

    im1 = axes[0, 1].imshow(true_grid, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    axes[0, 1].set_title("Reference surface")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.9)

    im2 = axes[1, 0].imshow(err_grid, origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
    axes[1, 0].set_title("Absolute error")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("t")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.9)

    t_slice_idx = len(t_val) // 2
    t_slice = t_val[t_slice_idx]
    axes[1, 1].plot(x_val, pred_grid[t_slice_idx, :], label="Prediction", color="tab:blue")
    axes[1, 1].plot(x_val, true_grid[t_slice_idx, :], label="Reference", color="tab:orange", linestyle="--")
    axes[1, 1].set_title(f"Slice at t = {t_slice:.2f}")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("u(x, t)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Wave PINN solution at epoch {epoch}")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = output_dir / f"wave_solution_epoch_{epoch}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    model.train()


def Run_wave_pinn():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"使用Device: {device}")

    wave_net = FullyConnected(
        in_features=2,
        out_features=1,
        num_layers=6,
        layer_size=128,
        activation_fn="tanh",
    ).to(device)

    L = np.pi
    T = 2.0 * np.pi
    c = 1.0

    num_interior = 4096
    num_ic = 2048
    num_bc = 2048

    dx, dt = 0.02, 0.02
    x_val = np.arange(0, L, dx)
    t_val = np.arange(0, T, dt)
    X_mesh, T_mesh = np.meshgrid(x_val, t_val)
    X_flat = X_mesh.reshape(-1, 1)
    T_flat = T_mesh.reshape(-1, 1)

    u_true = np.sin(X_flat) * (np.cos(T_flat) + np.sin(T_flat))

    val_input = torch.tensor(
        np.concatenate([X_flat, T_flat], axis=1),
        dtype=torch.float32,
        device=device,
    )
    val_true = torch.tensor(u_true, dtype=torch.float32, device=device)

    output_dir = Path(__file__).resolve().parent / "outputs" / "wave_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"结果图片将保存到: {output_dir}")

    optimizer = torch.optim.Adam(wave_net.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    print("开始训练...")
    total_epochs = 3000
    for epoch in range(total_epochs):
        optimizer.zero_grad()

        x_int = torch.rand(num_interior, 1, dtype=torch.float32, requires_grad=True, device=device) * L
        t_int = torch.rand(num_interior, 1, dtype=torch.float32, device=device) * T
        loss_pde = pde_loss(wave_net, x_int, t_int, c, mse_loss)

        x_ic = torch.rand(num_ic, 1, dtype=torch.float32, requires_grad=True, device=device) * L
        t_ic = torch.zeros(num_ic, 1, dtype=torch.float32, requires_grad=True, device=device)
        u_ic_target = torch.sin(x_ic).detach()
        ut_ic_target = torch.sin(x_ic).detach()

        input_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic_pred = wave_net(input_ic)
        loss_ic = mse_loss(u_ic_pred, u_ic_target)

        u_ic_pred_t = torch.autograd.grad(
            u_ic_pred,
            t_ic,
            grad_outputs=torch.ones_like(u_ic_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        loss_ic_ut = mse_loss(u_ic_pred_t, ut_ic_target)

        t_bc = torch.rand(num_bc, 1, dtype=torch.float32, device=device) * T
        x_bc_left = torch.zeros(num_bc, 1, dtype=torch.float32, device=device)
        x_bc_right = L * torch.ones(num_bc, 1, dtype=torch.float32, device=device)

        input_bc_left = torch.cat([x_bc_left, t_bc], dim=1)
        input_bc_right = torch.cat([x_bc_right, t_bc], dim=1)
        u_bc_l_pred = wave_net(input_bc_left)
        u_bc_r_pred = wave_net(input_bc_right)
        loss_bc = mse_loss(u_bc_l_pred, torch.zeros_like(u_bc_l_pred)) + mse_loss(u_bc_r_pred, torch.zeros_like(u_bc_r_pred))

        total_loss = loss_pde + loss_ic + loss_ic_ut + loss_bc
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{total_epochs}], Total Loss: {total_loss.item():.6f}")

            wave_net.eval()
            with torch.no_grad():
                u_val_pred = wave_net(val_input)
                val_loss = mse_loss(u_val_pred, val_true)
                print(f"Validation Loss: {val_loss.item():.6f}")
            wave_net.train()

            save_solution_plots(epoch + 1, wave_net, X_mesh, x_val, t_val, val_input, val_true, output_dir)


if __name__ == "__main__":
    Run_wave_pinn()
   