# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp.fully_connected import FullyConnected
from sympy import Function, Number, Symbol

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.planar.structured_grid import (
    load as load_structured_grid,
)
from physicsnemo.mesh.sampling import sample_random_points_on_cells
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, StaticCaptureTraining
from omegaconf import DictConfig
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

# 二维不可压缩 Navier-Stokes 方程 (steady, 2D)
class NavierStokes(PDE):
    """Incompressible Navier-Stokes equations (steady, 2D).

    Simplified from the compressible form in physicsnemo-sym for the case
    where ``rho`` is constant and ``time=False``.

    Reference: https://turbmodels.larc.nasa.gov/implementrans.html
    """

    def __init__(self, nu=0.01, rho=1.0, dim=2, time=False):
        self.dim = dim
        x, y = Symbol("x"), Symbol("y")
        iv = {"x": x, "y": y}
        u = Function("u")(*iv.values())
        v = Function("v")(*iv.values())
        p = Function("p")(*iv.values())
        nu, rho = Number(nu), Number(rho)
        self.equations = {
            # 二维连续性方程
            "continuity": u.diff(x) + v.diff(y),
            # 动量方程
            "momentum_x": (
                u * u.diff(x)
                + v * u.diff(y)
                + (1 / rho) * p.diff(x)
                - nu * u.diff(x, 2)
                - nu * u.diff(y, 2)
            ),
            "momentum_y": (
                u * v.diff(x)
                + v * v.diff(y)
                + (1 / rho) * p.diff(y)
                - nu * v.diff(x, 2)
                - nu * v.diff(y, 2)
            ),
        }

# Hydra 是一个 Python 配置管理库，常用于把训练参数、模型配置、实验设置从代码里“解耦”出来，方便切换不同实验配置。
# @hydra.main(...)：让函数自动接收这个配置对象
@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def ldc_trainer(cfg: DictConfig) -> None:
    """Main function for the LDC PINNs."""

    # 准备分布式训练环境，初始化分布式管理器.这个管理器同一处理设备选择和分布式上下文(cpu,gpu)
    # dist.device的作用:有GPU就用GPU,没有就用CPU,并且在分布式训练中,dist.device会根据rank选择不同的GPU
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="ldc")
    log.file_logging()  # 开启文件日志记录,将日志写入文件

    # domain geometry using physicsnemo.mesh
    height = 0.1
    width = 0.1
    x_min, x_max = -width / 2, width / 2
    y_min, y_max = -height / 2, height / 2

    # load_structured_grid: 加载一个结构化网格,用于定义流体域的几何形状
    interior_mesh = load_structured_grid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n_x=50,
        n_y=50,
        device=dist.device,
    )
    # 获取边界网格,用于定义流体域的边界条件
    boundary_mesh = interior_mesh.get_boundary_mesh()

    # 定义采样函数,用于在边界和内部采样点
    def sample_boundary(n_points, device):
        """Sample on the rectangle boundary using physicsnemo.mesh."""
        cell_indices = torch.randint(
            0, boundary_mesh.n_cells, (n_points,), device=device
        )
        pts = sample_random_points_on_cells(boundary_mesh, cell_indices)
        return {"x": pts[:, 0], "y": pts[:, 1]}

    def sample_interior(n_points, device):
        """Sample inside the rectangle using physicsnemo.mesh, with analytical SDF."""
        cell_indices = torch.randint(
            0, interior_mesh.n_cells, (n_points,), device=device
        )
        pts = sample_random_points_on_cells(interior_mesh, cell_indices)
        x, y = pts[:, 0], pts[:, 1]
        # 把 [x - x_min, x_max - x, y - y_min, y_max - y] 这 4 个张量堆叠起来
        # 然后在最后一个维度上取最小值,得到每个点到边界的最短距离,即SDF
        # .values 是因为 torch.min 返回的是一个 namedtuple,包含 values 和 indices,我们只需要 values
        sdf = torch.min(
            torch.stack([x - x_min, x_max - x, y - y_min, y_max - y], dim=-1),
            dim=-1,
        ).values    
        return {"x": x, "y": y, "sdf": sdf}

    # 全连接神经网络模型,输入2维(x,y),输出3维(u,v,p), 隐藏层6层,每层512个神经元
    model = FullyConnected(
        in_features=2, out_features=3, num_layers=6, layer_size=512
    ).to(dist.device)

    # 定义物理信息神经网络的约束条件,使用NavierStokes类定义PDE方程,并使用PhysicsInformer计算PDE残差
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    phy_inf = PhysicsInformer(
        required_outputs=["continuity", "momentum_x", "momentum_y"],
        equations=ns,
        grad_method="autodiff",
        device=dist.device,
    )

    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    # 学习率衰减,让优化器的学习率随着训练步数增加而逐渐减小,以便在训练后期更稳定地收敛
    # 因为训练前期通常需要较大学习率，快速下降；后期为了更稳定地收敛，通常要减小学习率。
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.9999871767586216**step
    )

    # 这里是规则均匀的网格，用于推理和可视化模型预测结果。
    # inference geometry
    x = np.linspace(-0.05, 0.05, 512)
    y = np.linspace(-0.05, 0.05, 512)
    # 二维网格坐标,xx存储x坐标,y存储y坐标,用于可视化模型预测结果
    xx, yy = np.meshgrid(x, y, indexing="xy")
    # 转换为tensor,并放到指定设备上(dist.device),用于后续模型推理
    xx, yy = (
        torch.from_numpy(xx).to(torch.float).to(dist.device),
        torch.from_numpy(yy).to(torch.float).to(dist.device),
    )

    for i in range(5000):
        # 清空梯度,避免梯度累积,因为PyTorch默认会累积梯度,所以每次训练前都要清空梯度
        optimizer.zero_grad()
        
        # 采样边界点和内部点,用于计算PDE残差和边界条件损失
        bc_data = sample_boundary(2000, dist.device)
        int_data = sample_interior(4000, dist.device)

        y_vals = bc_data["y"]
        # 根据y坐标判断边界点是上边界还是其他边界,用于分别处理不同的边界条件，bool值Tensor对象
        mask_top_wall = y_vals >= height / 2 - 1e-7     # 上边界条件,速度为1
        mask_no_slip = ~mask_top_wall                   # 其他边界条件,速度为0

        # 除了上边界的边界点
        # 使用之前的bool值Tensor对象,从边界点中筛选
        no_slip_xy = torch.stack(
            [bc_data["x"][mask_no_slip], bc_data["y"][mask_no_slip]], dim=-1
        )
        # 上边界点的x坐标
        top_wall_x = bc_data["x"][mask_top_wall].unsqueeze(-1)  # 改变为[N,1]形状,用于计算上边界条件损失
        # 上边界点的(x,y)坐标, 改为[N,2]形状,用于计算上边界条件损失
        top_wall_xy = torch.stack(
            [bc_data["x"][mask_top_wall], bc_data["y"][mask_top_wall]], dim=-1
        )

        # 内部点的(x,y)坐标, 改为[N,2]形状,用于计算PDE残差
        int_x = int_data["x"].unsqueeze(-1).requires_grad_(True)    #
        int_y = int_data["y"].unsqueeze(-1).requires_grad_(True)
        int_sdf = int_data["sdf"].unsqueeze(-1)
        coords = torch.cat([int_x, int_y], dim=1)

        # 模型前向传播,计算边界点和内部点的预测值
        no_slip_out = model(no_slip_xy)
        top_wall_out = model(top_wall_xy)
        interior_out = model(coords)

        # 这里直接计算均方误差,没有使用mse_criterion = nn.MSELoss()...这种方式
        u_no_slip = torch.mean(no_slip_out[:, 0:1] ** 2)    # 其它边界u loss
        v_no_slip = torch.mean(no_slip_out[:, 1:2] ** 2)    # 其它变价v loss
        # 上边界loss, 上边界速度要接近 1，但边缘位置可以放松一点”，通常是为了减少角点附近的训练不稳定。
        # 目标速度是 u=1，所以先用平方误差(u-1)^2,再乘以空间权重(1-20|X|),这个权重在边界中间最大，在左右两端接近 0
        u_slip = torch.mean(
            ((top_wall_out[:, 0:1] - 1.0) ** 2) * (1 - 20 * torch.abs(top_wall_x))
        )
        v_slip = torch.mean(top_wall_out[:, 1:2] ** 2)  # 上边界v loss, v=0

        #把坐标、u、v、p 这四个张量打包成一个字典，交给 phy_inf.forward 去计算 PDE 残差
        phy_loss_dict = phy_inf.forward(
            {
                "coordinates": coords,
                "u": interior_out[:, 0:1],
                "v": interior_out[:, 1:2],
                "p": interior_out[:, 2:3],
            }
        )
        # 给内部点的 PDE 残差加权，而不是让每个内部采样点一视同仁。
        cont = phy_loss_dict["continuity"] * int_sdf        # 连续性方程
        mom_x = phy_loss_dict["momentum_x"] * int_sdf       # 动量方程x方向
        mom_y = phy_loss_dict["momentum_y"] * int_sdf       # 动量方程y方向

        # 物理损失函数,包括 PDE 残差和边界条件损失
        phy_loss = (
            torch.mean(cont**2)
            + torch.mean(mom_x**2)
            + torch.mean(mom_y**2)
            + u_no_slip
            + v_no_slip
            + u_slip
            + v_slip
        )
        phy_loss.backward() # 反向传播，计算梯度
        optimizer.step()    # 更新权重参数
        scheduler.step()    # 更新学习率

        if i % 200 == 0:
            print(f"Step {i} / 5000, Loss: {phy_loss.detach()}, LR: {optimizer.param_groups[0]['lr']}")

            # 保存当前的模型权重到文件,用于后续推理或继续训练
            torch.save(
                model.state_dict(), f"./outputs/model_weights_{i}.pth"
            )

            with torch.no_grad():
                # 组合为[N,2]形状的坐标张量,用于模型推理
                inf_out = model(
                    torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
                )
                # print(
                #     f"Loss: {phy_loss.detach()}, LR: {optimizer.param_groups[0]['lr']}"
                # )
                fig, axes = plt.subplots(1, 4, figsize=(12, 4))
                # 将模型输出从 GPU 转移到 CPU，并转换为 NumPy 数组,用于可视化
                out_np = inf_out.detach().cpu().numpy()
                xx_np = xx.detach().cpu().numpy()
                yy_np = yy.detach().cpu().numpy()
                # 可视化模型预测结果,包括速度u、速度v、压力p和速度大小
                im = axes[0].imshow(out_np[:, 0].reshape(512, 512), origin="lower")
                fig.colorbar(im, ax=axes[0])
                axes[0].set_title("u")

                im = axes[1].imshow(out_np[:, 1].reshape(512, 512), origin="lower")
                fig.colorbar(im, ax=axes[1])
                axes[1].set_title("v")

                im = axes[2].imshow(out_np[:, 2].reshape(512, 512), origin="lower")
                fig.colorbar(im, ax=axes[2])
                axes[2].set_title("p")

                im = axes[3].imshow(
                    ((out_np[:, 0] ** 2 + out_np[:, 1] ** 2).reshape(512, 512)) ** 0.5,
                    origin="lower",
                )
                fig.colorbar(im, ax=axes[3])
                axes[3].set_title("u_mag")

                plt.savefig(f"./outputs/outputs_pc_{i}.png")
                plt.close()

                # 可视化流线图,使用速度场的大小作为颜色映射
                u_grid = out_np[:, 0].reshape(512, 512)
                v_grid = out_np[:, 1].reshape(512, 512)
                speed_grid = np.sqrt(u_grid**2 + v_grid**2)

                fig_stream, ax_stream = plt.subplots(figsize=(5, 5))
                stream = ax_stream.streamplot(
                    xx_np,
                    yy_np,
                    u_grid,
                    v_grid,
                    color=speed_grid,
                    cmap="viridis",
                    density=1.5,
                    linewidth=1,
                )
                fig_stream.colorbar(stream.lines, ax=ax_stream, label="|u|")
                ax_stream.set_title(f"Streamlines at step {i}")
                ax_stream.set_xlabel("x")
                ax_stream.set_ylabel("y")
                ax_stream.set_aspect("equal")
                fig_stream.tight_layout()
                fig_stream.savefig(f"./outputs/streamlines_{i}.png")
                plt.close(fig_stream)


if __name__ == "__main__":
    ldc_trainer()
