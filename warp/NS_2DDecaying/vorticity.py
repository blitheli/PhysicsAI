import numpy as np
import warp as wp
from utils import validate_diffusion,validate_advection,validate_fft_roundtrip,initialize_decaying_turbulence
from PIL import Image
from matplotlib.colors import Normalize
import os

#########################################################
#            2-D 涡量
#########################################################
if wp.get_cuda_device_count() > 0:
    print("GPU 存在！")
else:
    raise RuntimeError("No CUDA-capable GPU detected.")

# 初始化
N_GRID = 512
LEN =  2 *np.pi

# 步长
DT =  0.0005
# 雷诺数
RE = 1000.0
# 网格常数
H = LEN / N_GRID

# 涡量 timestep n
omega_0 = wp.zeros((N_GRID, N_GRID), dtype=wp.float32)
# 涡量 timestep n+1
omega_1 = wp.zeros_like(omega_0)

# 流场函数psi
psi = wp.zeros((N_GRID, N_GRID), dtype=wp.float32)

# 检查数组是否正确分配
for name, arr in [("omega_0", omega_0), ("omega_1", omega_1), ("psi", psi)]:
    assert arr is not None, f"{name} is still None - fill in the allocation"
    assert isinstance(arr, wp.array), f"{name} should be a wp.array, got {type(arr)}"
    assert arr.shape == (N_GRID, N_GRID), f"{name}.shape = {arr.shape}, expected ({N_GRID}, {N_GRID})"
    assert arr.dtype == wp.float32, f"{name}.dtype = {arr.dtype}, expected float"

print("All arrays allocated correctly")


@wp.func
def cyclic_index(i: int, N: int) -> int:
    """返回给定索引在周期性边界条件下的循环索引。"""
    ret_idx = i % N
    if ret_idx < 0:
        ret_idx += N
    return ret_idx

@wp.kernel
def diffusion_kernel(omega: wp.array2d[float], laplacian: wp.array2d[float]):
    """计算涡量的扩散项。使用五点拉普拉斯算子模板。"""
    i, j = wp.tid()

    # 当前点的周围四个邻居的索引，使用周期性边界条件
    i_left = cyclic_index(i - 1, N_GRID)
    i_right = cyclic_index(i + 1, N_GRID)
    j_down = cyclic_index(j - 1, N_GRID)
    j_top = cyclic_index(j + 1, N_GRID)

    # 计算拉普拉斯算子, 涡量
    laplacian[i, j] = (
        omega[i_left, j]
        + omega[i_right, j]
        + omega[i, j_top]
        + omega[i, j_down]
        - 4.0 * omega[i, j]
    ) / (H * H)

@wp.kernel
def advection_kernel(omega: wp.array2d[float], psi: wp.array2d[float], advected_out: wp.array2d[float]):
    """计算涡量的对流项 J(psi, omega)。
    
    Args:
        omega (wp.array2d[float]): 涡量场
        psi (wp.array2d[float]): 流场函数
        advected_out (wp.array2d[float]): 输出的 advected 涡量

    """
    i, j = wp.tid()

    # 当前点的周围四个邻居的索引，使用周期性边界条件
    i_left = cyclic_index(i - 1, N_GRID)
    i_right = cyclic_index(i + 1, N_GRID)
    j_down = cyclic_index(j - 1, N_GRID)
    j_top = cyclic_index(j + 1, N_GRID)

    # 计算速度分量 u 和 v
    u = (psi[i, j_top] - psi[i, j_down]) / (2.0 * H)  # u = d(psi)/dy
    v = -(psi[i_right, j] - psi[i_left, j]) / (2.0 * H)  # v = -d(psi)/dx

    # 计算涡量的对流项
    advected_out[i, j] = u * (omega[i_right, j] - omega[i_left, j]) / (2.0 * H) + v * (omega[i, j_top] - omega[i, j_down]) / (2.0 * H)

@wp.kernel
def viscous_advection_kernel(omega_0: wp.array2d[float], psi: wp.array2d[float], omega_1: wp.array2d[float]):
    """计算涡量的粘性对流项，结合扩散和对流。
    
    Args:
        omega_0 (wp.array2d[float]): 当前涡量场
        psi (wp.array2d[float]): 流函数
        omega_1 (wp.array2d[float]): 输出的更新后的涡量场
    """
    i, j = wp.tid()

    # 当前点的周围四个邻居的索引，使用周期性边界条件
    i_left = cyclic_index(i - 1, N_GRID)
    i_right = cyclic_index(i + 1, N_GRID)
    j_down = cyclic_index(j - 1, N_GRID)
    j_top = cyclic_index(j + 1, N_GRID)

    # 计算拉普拉斯算子
    laplacian = (
        omega_0[i_left, j]
        + omega_0[i_right, j]
        + omega_0[i, j_top]
        + omega_0[i, j_down]
        - 4.0 * omega_0[i, j]
    ) / (H * H)


    # 计算速度分量 u 和 v
    u = (psi[i, j_top] - psi[i, j_down]) / (2.0 * H)  # u = d(psi)/dy
    v = -(psi[i_right, j] - psi[i_left, j]) / (2.0 * H)  # v = -d(psi)/dx

    # 对流项
    advected_term = u * (omega_0[i_right, j] - omega_0[i_left, j]) / (2.0 * H) + v * (omega_0[i, j_top] - omega_0[i, j_down]) / (2.0 * H)

    # 更新涡量
    omega_1[i, j] = omega_0[i, j] + DT * (laplacian/RE - advected_term)

@wp.kernel
def copy_float_to_complex(omega: wp.array2d[float], omega_complex: wp.array2d[wp.vec2f]):
    """将实数数组复制到复数数组中，实部为原始值，虚部为零。"""
    i, j = wp.tid()
    omega_complex[i, j] = wp.vec2f(omega[i, j], 0.0)

@wp.kernel
def fft_tiled(x: wp.array2d[wp.vec2f], y: wp.array2d[wp.vec2f]):
    """执行 一维傅里叶变换，使用分块方法。
    
    Args:
        x (wp.array2d[wp.vec2f]): 输入复数数组(2D)
        y (wp.array2d[wp.vec2f]): 输出复数数组(2D)
    """
    i = wp.tid()

    # 从输入数组中加载一个 tile
    a = wp.tile_load(x, shape=(1, N_GRID), offset=(i, 0))

    # 对 tile 执行 FFT
    wp.tile_fft(a)
    wp.tile_store(y, a, offset=(i, 0))

@wp.kernel
def ifft_tiled(x: wp.array2d[wp.vec2f], y: wp.array2d[wp.vec2f]):
    """执行 一维逆傅里叶变换，使用分块方法。
    
    Args:
        x (wp.array2d[wp.vec2f]): 输入复数数组(2D)
        y (wp.array2d[wp.vec2f]): 输出复数数组(2D)
    """
    i = wp.tid()

    # 从输入数组中加载一个 tile
    a = wp.tile_load(x, shape=(1, N_GRID), offset=(i, 0))

    # 对 tile 执行 IFFT
    wp.tile_ifft(a)
    wp.tile_store(y, a, offset=(i, 0))

@wp.kernel
def transpose(x: wp.array2d[wp.vec2f], y: wp.array2d[wp.vec2f]):
    """转置二维数组。
    
    Args:
        x (wp.array2d[wp.vec2f]): 输入复数数组(2D)
        y (wp.array2d[wp.vec2f]): 输出复数数组(2D)
    """
    i, j = wp.tid()

    y[j, i] = x[i, j]
    

@wp.kernel
def extract_real_and_scale(scale: float, complex_array: wp.array2d[wp.vec2f], real_array: wp.array2d[float]):
    """提取复数数组的实部并进行缩放(除以缩放因子)。

    Args:
        scale (float): 缩放因子
        complex_array (wp.array2d[wp.vec2f]): 输入复数数组(2D)
        real_array (wp.array2d[float]): 输出实数数组(2D)
    """
    i, j = wp.tid()

    real_array[i, j] = complex_array[i, j].x / scale

@wp.kernel
def multiply_k2_inverse(inv_k_sq: wp.array2d[float], omega_hat: wp.array2d[wp.vec2f], psi_hat: wp.array2d[wp.vec2f]):
    """求解泊松方程，在傅里叶空间中将 omega_hat 乘以 1/k^2 得到 psi_hat。

    Args:
        inv_k_sq (wp.array2d[float]): 预计算的 1/k^2 数组(2D)
        omega_hat (wp.array2d[wp.vec2f]): 输入的涡量傅里叶系数(2D)
        psi_hat (wp.array2d[wp.vec2f]): 输出的流函数傅里叶系数(2D)
    """
    i, j = wp.tid()

    psi_hat[i, j] = omega_hat[i, j] * inv_k_sq[i, j]

omega_complex = wp.zeros((N_GRID, N_GRID), dtype=wp.vec2f)
fft_temp_1 = wp.zeros((N_GRID, N_GRID), dtype=wp.vec2f)
fft_temp_2 = wp.zeros((N_GRID, N_GRID), dtype=wp.vec2f)
fft_temp_3 = wp.zeros((N_GRID, N_GRID), dtype=wp.vec2f)


# 返回离散傅里叶空间中的波数网格(频率坐标)k=[0,1,2,...,N/2-1, -N/2, ..., -2, -1]*1/L
k = np.fft.fftfreq(N_GRID, d = 1.0/N_GRID)
kx,ky = np.meshgrid(k, k)
k2 = kx**2 + ky**2
inv_k_sq_np = np.zeros_like(k2)
nozero = k2 != 0
inv_k_sq_np[nozero] = 1.0 / k2[nozero]
inv_k_sq_np = inv_k_sq_np.astype(np.float32)

# 转换为 Warp 数组
inv_k_sq = wp.array(inv_k_sq_np, dtype=float)

def advance_vorticity_by_dt(omega_0, omega_1, psi):
    """涡度场和流函数前进一个时间步"""
    # === Building Block 1: Solve equation 1 to update vorticity ===
    # 对于每个网格点，计算涡量的更新值，使用有限差分方法近似偏导数

    wp.launch(viscous_advection_kernel, dim=(N_GRID, N_GRID), inputs=[omega_0, psi, omega_1])

    # === Building Block 2: Solve equation 2 using FFT ===
    # 2-D 傅里叶变换以获得 Fourier 空间中的涡量
    # 求解泊松方程在 Fourier 空间中
    # 2-D 逆傅里叶变换以获得物理空间中的流场函数psi
    # 涡度场转换为复数
    wp.launch(copy_float_to_complex, dim=(N_GRID, N_GRID), inputs=[omega_1, omega_complex])
    # 对复数数组进行行方向的 FFT, row FFT -> transpose -> row FFT
    wp.launch_tiled(fft_tiled, dim=(N_GRID, 1), inputs=[omega_complex, fft_temp_1], block_dim=N_GRID // 2)
    wp.launch(transpose, dim=(N_GRID, N_GRID), inputs=[fft_temp_1, fft_temp_2])
    wp.launch_tiled(fft_tiled, dim=(N_GRID, 1), inputs=[fft_temp_2, fft_temp_3], block_dim=N_GRID // 2)
    # 傅里叶空间中乘以 1/k^2
    wp.launch(multiply_k2_inverse, dim=(N_GRID, N_GRID), inputs=[inv_k_sq, fft_temp_3, fft_temp_1])

    # 逆傅里叶变换以获得物理空间中的流函数psi
    wp.launch_tiled(ifft_tiled, dim=(N_GRID, 1), inputs=[fft_temp_1, fft_temp_2], block_dim=N_GRID // 2)
    wp.launch(transpose, dim=(N_GRID, N_GRID), inputs=[fft_temp_2, fft_temp_3])
    wp.launch_tiled(ifft_tiled, dim=(N_GRID, 1), inputs=[fft_temp_3, fft_temp_1], block_dim=N_GRID // 2)

    # 提取实部
    wp.launch(extract_real_and_scale, dim=(N_GRID, N_GRID), inputs=[float(N_GRID*N_GRID), fft_temp_1, psi])

    # 复制 omega_1 到 omega_0
    wp.copy(omega_0, omega_1)
    


print("------------验证扩散项------------------------")
validate_diffusion(diffusion_kernel, n_grid=512, kx = 2, ky =3)
validate_advection(advection_kernel, n_grid=512, kx = 2, ky =3)
print("---------------------------------------------")
print("------------验证 FFT roundtrip ------------------------")
fig, axes, max_error = validate_fft_roundtrip(
    fft_kernel=fft_tiled,
    ifft_kernel=ifft_tiled,
    n_grid=N_GRID,
    tile_m=1,
    tile_n=N_GRID,
    block_dim=N_GRID // 2,
)
print(f"FFT roundtrip max error: {max_error:.3e}")
print("---------------------------------------------")
print("  plot 1/k2  ")
# Visualize the precomputed field using a headless backend so this script can run
# in a server or container without a GUI display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(np.log10(inv_k_sq_np + 1e-12), cmap="viridis", origin="lower")
fig.colorbar(im, ax=ax, label=r"$\log_{10}(1/|k|^2)$")
ax.set_title(r"Precomputed $1/|k|^2$ (log scale)")
fig.tight_layout()
fig.savefig("inv_k_sq_log.png", dpi=150)
plt.close(fig)
print("Saved visualization to inv_k_sq_log.png")
print("---------------------------------------------")

# “按照湍流的能谱分布，随机生成一个合理的初始 2D 涡量场”，作为后面时间推进的起始状态。
omega_init_np = initialize_decaying_turbulence(n_grid=N_GRID, seed = 42)
omega_0 = wp.array(omega_init_np, dtype=float)
omega_1 = wp.zeros_like(omega_0)
# 二维离散傅里叶变换(2D FFT)
omega_hat = np.fft.fft2(omega_init_np)
psi_init_np = np.fft.ifft2(omega_hat * inv_k_sq_np).real.astype(np.float32)
psi = wp.array(psi_init_np, dtype=float)

# 使用CUDA Graph 来捕获一次 advance_vorticity_by_dt 的执行，以便后续重复调用时可以更高效地执行。
with wp.ScopedCapture() as capture:
    advance_vorticity_by_dt(omega_0, omega_1, psi)
setp_graph = capture.graph

NUM_FRAMES= 400
STEPS_PER_FRAME = 20
GIF_SIZE = 512

cmap = plt.cm.twilight
norm = Normalize(vmin=-15, vmax=15)

frame_baseline = []
print(f"\n开始运行 {NUM_FRAMES} 帧的模拟, 每帧包含 {STEPS_PER_FRAME} 步")

for frame in range(NUM_FRAMES):
    for _ in range(STEPS_PER_FRAME):

        wp.capture_launch(setp_graph)  # 使用捕获的 CUDA Graph 来执行 advance_vorticity_by_dt

    # 将当前帧的涡量场保存到 frame_baseline 列表中
    vorticity = omega_1.numpy().T  # 转置以匹配物理空间的布局
    colored = cmap(norm(vorticity))
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    pil_frame = Image.fromarray(rgb).resize((GIF_SIZE, GIF_SIZE), Image.LANCZOS)
    frame_baseline.append(pil_frame)

    if (frame + 1) % 5 == 0:
        print(f"Completed frame {frame + 1}/{NUM_FRAMES}")

total_steps = NUM_FRAMES * STEPS_PER_FRAME


# Create animated GIF to visualize time evolution
print("-----------------------------------")
print("开始创建动画 GIF...")
os.makedirs("./images", exist_ok=True)
output_filename = (
    f"./images/turbulence_{GIF_SIZE}x{GIF_SIZE}.gif"
)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

# Save as animated GIF (100ms per frame = 10 FPS)
frame_baseline[0].save(
    output_filename,
    save_all=True,
    append_images=frame_baseline[1:],
    duration=100,  # milliseconds per frame
    loop=0,  # infinite loop
)
print("-----------------------------------------")




