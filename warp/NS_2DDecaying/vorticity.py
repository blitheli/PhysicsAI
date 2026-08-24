import numpy as np
import warp as wp

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
    

def advance_vorticity_by_dt(omega_0, omega_1, psi):
    """Placeholder for the vorticity advancement update."""
    # === Building Block 1: Solve equation 1 to update vorticity ===
    # 对于每个网格点，计算涡量的更新值，使用有限差分方法近似偏导数

    # === Building Block 2: Solve equation 2 using FFT ===
    # 2-D 傅里叶变换以获得 Fourier 空间中的涡量
    # 求解泊松方程在 Fourier 空间中
    # 2-D 逆傅里叶变换以获得物理空间中的流场函数psi
    return omega_1


a = wp.array([[1.0, 2.0, 3.0], [4, 5, 6], [7, 8, 9]], dtype=wp.float32)
b = wp.zeros(a.shape, dtype=wp.vec2f)
c = wp.zeros(a.shape, dtype=wp.vec2f)
wp.launch(copy_float_to_complex, dim=a.shape, inputs=[a, b])

wp.launch(transpose, dim=a.shape, inputs=[b, c])

cc = c.numpy()
print(cc)