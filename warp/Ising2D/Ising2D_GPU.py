import numpy as np
import warp as wp
import matplotlib.pyplot as plt

# Check for GPU availability
if wp.get_cuda_device_count() > 0:
    print("✓ GPU detected successfully")
else:
    raise RuntimeError(
        "No CUDA-capable GPU detected. This notebook assumes at least one NVIDIA GPU."
    )

@wp.kernel
def generate_lattice(lattice: wp.array2d[wp.int8], rng_seed: int):
    """
        初始化一个方形晶格，随机分配自旋方向。
        Args:
            lattice (wp.array2d[int]): 2-D array to store the spin configuration
            L (int): Linear size of the lattice (creates L×L grid)
    """

    # 这里使用 Warp 的线程索引来确定每个线程处理的晶格位置
    i, j = wp.tid() 

    # 使用 Warp 的随机数生成器，基于线程索引和传入的种子生成随机数
    thread_offset = i * lattice.shape[1] + j
    rng_state = wp.rand_init(rng_seed, thread_offset)  

    # 随机分配自旋方向
    lattice[i, j] = wp.int8(1) if wp.randf(rng_state, 0.0, 1.0) < 0.5 else wp.int8(-1)

@wp.kernel
def combine_lattices(lattice_b: wp.array2d[wp.int8], lattice_w: wp.array2d[wp.int8], lattice_out: wp.array2d[wp.int8]):
    """
        将两个晶格合并为一个输出晶格。
        Args:
            lattice_b (wp.array2d[int]): 黑色晶格 N*N/2
            lattice_w (wp.array2d[int]): 白色晶格 N*N/2
            lattice_out (wp.array2d[int]): 输出晶格
    """
    i,j =  wp.tid()
    if(i%2==0):
        lattice_out[i,2*j] = lattice_w[i,j]
        lattice_out[i,2*j+1] = lattice_b[i,j]
    else:
        lattice_out[i,2*j] = lattice_b[i,j]
        lattice_out[i,2*j+1] = lattice_w[i,j]

@wp.kernel
def update_lattice(beta: float, rng_seed: int, is_black: bool, op_lattice: wp.array2d[wp.int8], lattice: wp.array2d[wp.int8]):
    """
        更新晶格自旋配置，使用 Metropolis 算法。
        Args:
            beta (float): 逆温度参数
            rng_seed (int): 随机数种子
            is_black (bool): 是否更新黑色晶格
            op_lattice (wp.array2d[int]): 原始晶格
            lattice (wp.array2d[int]): 更新后的晶格
    """
    i,j = wp.tid()

    lattice_size = lattice.shape[0]
    num_sub_cols = lattice_size // 2

    sub_col_left_idx = (j-1) % num_sub_cols
    sub_col_right_idx = (j+1) % num_sub_cols

    if is_black:

        if i%2 == 0:
            secondary_i = sub_col_right_idx
        else:
            secondary_i = sub_col_left_idx
    else:
        if i%2 == 0:
            secondary_i = sub_col_left_idx
        else:
            secondary_i = sub_col_right_idx

    # 计算当前自旋的邻居自旋之和，使用模运算处理边界条件，实现周期性边界
    nn_sum = (
        op_lattice[(i-1) % lattice_size, j] +  # 上
        op_lattice[(i+1) % lattice_size, j] +  # 下
        op_lattice[i, j] +            # 左
        op_lattice[i, secondary_i]    # 右
    )
    # 计算当前自旋的值 (+1 或 -1)
    spin_ij = lattice[i,j]

    accept_prob = wp.exp(-2.0 * beta * wp.float32(spin_ij * nn_sum))

    rng_state = wp.rand_init(rng_seed, i * num_sub_cols + j)
    random_value = wp.randf(rng_state, 0.0, 1.0)

    if random_value < accept_prob:
        lattice[i,j] = -spin_ij  # 翻转自旋


LATTICE_SIZE = 64

# 创建两个空的 Warp 2D 数组来存储黑色和白色晶格自旋配置
lattice_b =  wp.empty((LATTICE_SIZE, LATTICE_SIZE//2), dtype=wp.int8)
lattice_w =  wp.empty((LATTICE_SIZE, LATTICE_SIZE//2), dtype=wp.int8)

# 创建一个空的 Warp 2D 数组来存储晶格自旋配置
lattice = wp.empty((LATTICE_SIZE, LATTICE_SIZE), dtype=wp.int8)

lattice_w.fill_(1)  # 将白色晶格初始化为全 +1 自旋
lattice_b.fill_(-1)  # 将黑色晶格初始化为全 -1 自旋

# 使用 Warp 的 launch 方法并行执行 generate_lattice 内核函数，初始化晶格
wp.launch(combine_lattices, lattice_b.shape, inputs=[lattice_b, lattice_w, lattice])

fig = plt.figure()
plt.imshow(lattice.numpy(), cmap="viridis")
plt.title("Grid Spin Configuration")
plt.savefig("./images/Checkerboard_lattice.png", dpi=300, bbox_inches="tight")
plt.close(fig)
#-----------------------------------------------------------------

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import IPython.display

import os
import time

LATTICE_SIZE = 256
TEMPERATURE = 2.0  # 临界温度
BETA = 1.0 / TEMPERATURE  # 逆温度 β = 1/(k_B * T)，这里假设 k_B=1

lattice_b = wp.empty((LATTICE_SIZE, LATTICE_SIZE//2), dtype=wp.int8)
lattice_w = wp.empty((LATTICE_SIZE, LATTICE_SIZE//2), dtype=wp.int8)
comb_lattices = wp.empty((LATTICE_SIZE, LATTICE_SIZE), dtype=wp.int8)

print(f"模拟初始条件，{LATTICE_SIZE}×{LATTICE_SIZE} Ising model at T={TEMPERATURE}")

# 生成黑白晶格 初始状态
wp.launch(generate_lattice, lattice_b.shape, inputs=[lattice_b, 42])
wp.launch(generate_lattice, lattice_w.shape, inputs=[lattice_w, 42])

# 使用 Viridis colormap 映射自旋值到颜色
viridis = plt.cm.viridis  
# 归一化自旋值到 [0, 1] 范围, -1 映射到 0, +1 映射到 1
norm = Normalize(vmin=-1, vmax=1)  

flames = []

start_time = time.perf_counter()

# 运行 200 步蒙特卡洛模拟, 每步更新晶格自旋配置(kenel)并保存当前帧用于动画
for step in range(200):

    # 更新所有黑色晶格自旋配置
    # 使用kernel 执行一次蒙特卡洛步长，更新晶格自旋配置
    wp.launch(update_lattice, (LATTICE_SIZE, LATTICE_SIZE//2), inputs=[BETA, step + 17,  True, lattice_w, lattice_b])

    # 更新所有白色晶格自旋配置
    wp.launch(update_lattice, (LATTICE_SIZE, LATTICE_SIZE//2), inputs=[BETA, step + 17, False, lattice_b, lattice_w])

    # 将黑白晶格合并为一个完整的晶格，用于可视化
    wp.launch(combine_lattices, lattice_b.shape, inputs=[lattice_b, lattice_w, comb_lattices])


    # 将晶格自旋值映射到颜色, comb_lattices.numpy() 将 Warp 数组转换为 NumPy 数组,因此np.array可以省略
    colored_frame = viridis(norm(np.array(comb_lattices.numpy()))) 

    rgb_frame = (colored_frame[:, :, :3] * 255).astype(np.uint8)  # 转换为 8-bit RGB
    flames.append(rgb_frame)  # 保存当前帧

    if (step + 1) % 10 == 0:
        total_spin = np.sum(comb_lattices.numpy())
        magnetization = total_spin / (LATTICE_SIZE * LATTICE_SIZE)
        print(f"  Step {step + 1}/200, 当前磁化强度 M = {magnetization:.4f}")

end_time = time.perf_counter()
print(f"模拟耗时 {end_time - start_time:.2f} 秒")

# Create animated GIF to visualize time evolution
print("开始创建动画 GIF...")
os.makedirs("./images", exist_ok=True)
pil_images = [Image.fromarray(frame) for frame in flames]
output_filename = (
    f"./images/warp_checkedboard_{LATTICE_SIZE}x{LATTICE_SIZE}_{TEMPERATURE}.gif"
)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

# Save as animated GIF (100ms per frame = 10 FPS)
pil_images[0].save(
    output_filename,
    save_all=True,
    append_images=pil_images[1:],
    duration=100,  # milliseconds per frame
    loop=0,  # infinite loop
)


