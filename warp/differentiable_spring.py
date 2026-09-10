# pyright: reportInvalidTypeForm=false
"""使用 wp.Tape() 通过终点位置反推弹簧振子的初始速度。

运行：python differentiable_spring.py
"""

import numpy as np
import warp as wp


wp.init()

DT = 0.01
NUM_STEPS = 100
SPRING_STIFFNESS = 4.0
MASS = 1.0
TARGET_POSITION = 0.75
LEARNING_RATE = 0.8
NUM_ITERATIONS = 20


@wp.kernel
def initialize_state(
    initial_velocity: wp.array(dtype=float),
    position: wp.array(dtype=float),
    velocity: wp.array(dtype=float),
):
    """将质点置于原点，并把待优化参数写入速度状态。"""
    position[0] = 0.0
    velocity[0] = initial_velocity[0]

"""
position、velocity：因为要被 kernel 改写，所以必须是数组。
"""

# 弹簧步进核函数：根据当前位置和速度计算下一步的状态。
@wp.kernel
def spring_step(
    position: wp.array(dtype=float),
    velocity: wp.array(dtype=float),
    next_position: wp.array(dtype=float),
    next_velocity: wp.array(dtype=float),
    dt: float,
    stiffness: float,
    mass: float,
):
    """半隐式 Euler：a = -k * x / m。"""
    acceleration = -stiffness * position[0] / mass
    next_velocity[0] = velocity[0] + dt * acceleration
    next_position[0] = position[0] + dt * next_velocity[0]

# 损失核函数：计算当前位置与目标位置的平方距离。
@wp.kernel
def squared_target_loss(
    position: wp.array(dtype=float), target: float, loss: wp.array(dtype=float)
):
    distance = position[0] - target
    loss[0] = distance * distance


def simulate(initial_velocity, device):
    """从初始状态正向模拟，并返回最后位置和标量损失。"""

    # 初始化位置和速度数组。
    position = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    velocity = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    wp.launch(initialize_state, dim=1, inputs=[initial_velocity, position, velocity], device=device)

    for _ in range(NUM_STEPS):
        next_position = wp.zeros_like(position, requires_grad=True)
        next_velocity = wp.zeros_like(velocity, requires_grad=True)
        wp.launch(
            spring_step,
            dim=1,
            inputs=[position, velocity, next_position, next_velocity, DT, SPRING_STIFFNESS, MASS],
            device=device,
        )
        position, velocity = next_position, next_velocity

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    wp.launch(squared_target_loss, dim=1, inputs=[position, TARGET_POSITION, loss], device=device)
    return position, loss


def main():
    device = wp.get_preferred_device()
    # 初始化初始速度数组。
    initial_velocity = wp.array([0.0], dtype=float, device=device, requires_grad=True)

    print(f"设备: {device}")
    print(f"目标: 在 t={NUM_STEPS * DT:.2f}s 时到达 x={TARGET_POSITION:.2f}\n")

    for iteration in range(NUM_ITERATIONS):
        # 录制并执行反向传播
        tape = wp.Tape()
        with tape:
            final_position, loss = simulate(initial_velocity, device)

        # 打印当前迭代的结果。
        tape.backward(loss)
        # 提取梯度、损失值和最终位置的数值。
        gradient = initial_velocity.grad.numpy()[0]
        loss_value = loss.numpy()[0]
        final_position_value = final_position.numpy()[0]

        print(
            f"迭代 {iteration:02d} | v0={initial_velocity.numpy()[0]: .5f} "
            f"| x(T)={final_position_value: .5f} | 损失={loss_value:.6f} "
            f"| dL/dv0={gradient: .5f}"
        )

        # 更新初始速度。
        updated_velocity = initial_velocity.numpy() - LEARNING_RATE * initial_velocity.grad.numpy()
        # 将更新后的速度赋值回初始速度，并清零梯度。
        initial_velocity.assign(updated_velocity)        
        initial_velocity.grad.zero_()
        # 重置反向传播 tape。
        # 原因是它会主动清掉这一轮 Tape 里录下来的计算图和相关引用，
        # 避免这些中间状态继续挂在对象上，尤其是设备内存和梯度关联状态。
        # 否则你就依赖 Python 垃圾回收何时真正回收旧 Tape。小例子里问题不大，迭代多了或图更大时，显式 reset 会更安全。
        tape.reset()

    print(f"\n优化后的初始速度: {initial_velocity.numpy()[0]:.5f}")


if __name__ == "__main__":
    main()