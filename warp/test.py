import warp as wp
import numpy as np

# 本质上是在用 Warp 对 100 万个粒子做一个简单的“朝原点吸引”的并行模拟。下面按行解释。

num_particles = 1000000
dt = 0.01

# 这不是普通的 Python 函数，而是会被 Warp 编译成可并行执行的计算单元
# 它通常不通过 return 返回结果，而是直接修改传入的数组。
# 它会被很多线程同时执行，每个线程通过后面的 wp.tid() 处理一个粒子。
@wp.kernel
def gravity_step(pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):

    i = wp.tid()    # 获取当前线程的索引，这里对应粒子的索引
    position = pos[i]
    dist_sq = wp.length_sq(position) + 0.01     # 避免除以零，计算距离平方
    acc= -1000.0/dist_sq* wp.normalize(position)    # 计算加速度，方向是朝向原点，大小与距离平方成反比
    vel[i] += acc*dt
    pos[i] += vel[i]*dt # 更新位置，使用简单的欧拉积分方法,这里不严谨，仅作演示

rng = np.random.default_rng(42)    
# 初始化粒子位置和速度，使用正态分布随机生成(1000000,3)的数组，并转换为 Warp 的 vec3 类型数组
positions = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3)    # 把这个numpy数组转换为Warp的vec3类型数组
velocities = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3)

for _ in range(100):

    # 这是整个程序真正执行并行计算的地方。
    # 调用 Warp 的 launch 方法，执行 gravity_step 内核函数，传入粒子位置和速度数组
    # dim表示粒子数量，这次launch会启动num_particles个线程，每个线程处理一个粒子。
    # 让100万个粒子在100步中不断被“吸引”到原点，模拟一个简单的重力场。
    wp.launch(gravity_step, dim=num_particles, inputs=[positions, velocities])

lastPos = positions.numpy()  # 将 Warp 的 vec3 类型数组转换回 numpy 数组，方便后续处理或可视化
print(lastPos.shape)  # 输出最后粒子位置的形状，应该是 (1000000, 3)