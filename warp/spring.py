# pyright: reportInvalidTypeForm=false
"""使用 wp.Tape() 通过终点位置反推弹簧振子的初始速度。

    弹簧振子运动公式: a = -k*x/m
    使用隐式欧拉法进行一步积分

    位置和速度为标量,但是为了兼容 wp.kernel 的输入输出要求，它们被包装成长度为 1 的数组。

运行: python spring_back.py
"""

import numpy as np
import warp as wp


wp.init()

DT = 0.01
NUM_STEPS = 100
STIFFNESS = 4.0
MASS = 1.0
TARGET_POSITION = -2.0
LEARNING_RATE = 0.85
NUM_ITERATIONS = 20


@wp.kernel
def initialize_state(
    vel0: wp.array(dtype=wp.float32),
    pos: wp.array(dtype=wp.float32),
    vel: wp.array(dtype=wp.float32),
):
    """初始化位置和速度状态。"""
    pos[0] = 0.0
    vel[0] = vel0[0]


@wp.kernel
def run_one_step(pos: wp.array(dtype=wp.float32), vel: wp.array(dtype=wp.float32), 
                 next_pos: wp.array(dtype=wp.float32), next_vel: wp.array(dtype=wp.float32), 
                 dt: float, k: float, mass: float):
    """执行弹簧系统的一步半隐式 Euler 积分。"""
    acc = -k * pos[0] / mass
    next_vel[0] = vel[0] + dt * acc
    next_pos[0] = pos[0] + dt * next_vel[0]

@wp.kernel
def get_loss(pos: wp.array(dtype=wp.float32), target: float, loss: wp.array(dtype=wp.float32)):
    """损失函数: 计算当前位置与目标位置的平方距离。"""
    distance = pos[0] - target
    loss[0] = distance * distance

def run(vel0: wp.array(dtype=wp.float32), device):
    """给定初始速度,正向积分, 返回最后位置和损失。"""

    # 赋初值
    pos = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    vel = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(kernel=initialize_state, dim=1, inputs=[vel0, pos, vel], device=device)

    # 循环走给定的步数
    for _ in range(NUM_STEPS):
        next_pos = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        next_vel = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        # 走一步 
        wp.launch(kernel=run_one_step, dim = 1, inputs=[pos, vel, next_pos, next_vel, DT, STIFFNESS, MASS], device=device)
        pos, vel = next_pos, next_vel

    # 计算最终的损失
    wp.launch(kernel=get_loss, dim = 1, inputs=[pos, TARGET_POSITION, loss], device=device)
    return pos, loss

def main():
    device = wp.get_cuda_device()
    print("设备: ", device)
    print(f"运行时长: {NUM_STEPS*DT} s, 最终到达目标位置 {TARGET_POSITION} m")
    # 初始速度为 0
    vel0 = wp.array([0.0], dtype=wp.float32, device=device, requires_grad=True)

    for iter in range(NUM_ITERATIONS):

        tap = wp.Tape()
        with tap:
            
            pos, loss = run(vel0, device)

        # 3. 反向传播：直接将标量数组传给 loss 参数
        # 这会自动把标量 loss 的梯度（dLoss/dLoss）隐式设为 1.0 并开始反向传播
        tap.backward(loss)

        # 读取梯度
        grad = vel0.grad.numpy()[0]
        loss_value = loss.numpy()[0]
        last_pos = pos.numpy()[0]

        # 打印当前循环状态                
        print(f"Iteration {iter}: v0 = {vel0.numpy()[0]:.5f}, loss = {loss_value: .5f}, grad = {grad: .5f}, last position = {last_pos: .5f}")

        # 更新初始速度
        updated_vel0 = vel0.numpy()
        updated_vel0[0] -= LEARNING_RATE * grad
        vel0.assign(updated_vel0)

        vel0.grad.zero_()
        tap.reset()

    # 打印更新后的初始速度
    print(f"最终的初始速度: {vel0.numpy()[0]: .5f}")


if __name__ == "__main__":
    main()


