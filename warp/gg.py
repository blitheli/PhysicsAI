import warp as wp
import numpy as np

num_particles = 35000
num_steps = 600

central_mass = 5000.0
softening = 0.5
dt = 0.005

def init_galaxy(num_particles, central_mass, disk_radius = 20.0, disk_thickness=0.5):
    """创建初始位置和速度数组, wp.array类型"""

    # 伪随机数生成器
    rng = np.random.default_rng(42)

    # 指数分布：中心极度密集，向外急剧变稀疏，平均值在1/3半径
    r = rng.exponential(scale = disk_radius/3.0, size = num_particles).astype(np.float32)

    # 旋转角度0-2pi均匀分布
    theta = rng.uniform(0, 2*np.pi, num_particles).astype(np.float32)

    # 增加螺旋臂扰动
    theta +=0.4*np.sin(2*theta+0.3*r)

    # 位置：在XZ平面上，Y方向有薄的垂直分布
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    # 垂直方向分布，中心越远，Y方向越窄，盘面中心厚一点，外圈越来越薄
    y = rng.normal(0, disk_thickness, num_particles).astype(np.float32) * (1.0 - r/(disk_radius*2))

    # 合并 
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    #velocities = np.zeros_like(positions)
    # Keplerian circular velocities: v = sqrt(GM/r), 切向速度, r<0.1时，速度不能过大
    v_circ = np.sqrt(central_mass / np.maximum(r, 0.1)) 
    vx = -v_circ * np.sin(theta)
    vz = v_circ * np.cos(theta)
    vy = np.zeros_like(vx)
    # 小的随机分散，增加真实感
    dispersion = 0.02
    vx+= rng.normal(0, dispersion*v_circ, num_particles)
    vy+= rng.normal(0, dispersion*v_circ*0.1, num_particles)
    vz+= rng.normal(0, dispersion*v_circ, num_particles)
    velocities = np.stack([vx, vy, vz], axis=1).astype(np.float32)

    return wp.array(positions, dtype=wp.vec3), wp.array(velocities, dtype=wp.vec3)

positions, velocities = init_galaxy(num_particles, central_mass)

@wp.kernel
def integrate_galaxy(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    central_mass: float,
    softening: float,
    dt: float,
):
    
    i = wp.tid()

    r = positions[i]
    dist_sq = wp.length_sq(r) + softening * softening
    inv_dist = 1.0 / wp.sqrt(dist_sq)
    acceleration = -central_mass * (inv_dist * inv_dist * inv_dist) * r
    velocity = velocities[i] + acceleration * dt

    # 采用隐式欧拉积分方法更新速度和位置，保持辛结构，避免能量漂移
    # 有震荡,但是能量不会漂移, 适合长时间模拟.短期内没有RK4之类的准确
    velocities[i] = velocity
    positions[i] = positions[i] + velocity * dt

# 运行模拟,每一步都调用integrate_galaxy内核函数
for step in range(num_steps):
    wp.launch(integrate_galaxy, dim=num_particles, inputs=[positions, velocities, central_mass, softening, dt])    

print(f"总粒子数: {num_particles}, 总步数: {num_steps}, 中心质量: {central_mass}, 软化参数: {softening}, 时间步长: {dt} ")
print(f"Final positions: {positions}")    


# 作图
#--------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from IPython.display import HTML

# Re-run the simulation, snapshotting positions to CPU every few steps
num_particles_viz = 35_000
num_steps_viz = 600
save_every = 5

positions, velocities = init_galaxy(num_particles_viz, central_mass)

trajectory = [positions.numpy()]
for step in range(num_steps_viz):
    wp.launch(
        integrate_galaxy,
        dim=num_particles_viz,
        inputs=[positions, velocities, central_mass, softening, dt],
    )
    if (step + 1) % save_every == 0:
        trajectory.append(positions.numpy())  # GPU -> CPU copy

# 把 3D 位置投影成 2D 视图，并且按径向距离给点上色，最后再叠一个步数文字标签。

# Tilted camera projection: 30 degrees from face-on, 20 degree azimuth rotation
cos_a, sin_a = np.cos(np.radians(20.0)), np.sin(np.radians(20.0))
cos_e, sin_e = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))

idx = np.arange(0, num_particles_viz, 2)  # display every other particle

fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#0e1117")
ax.set(xlim=(-25, 25), ylim=(-18, 18), aspect="equal")
ax.set_facecolor("#0e1117")
ax.axis("off")

pos0 = trajectory[0][idx]
xr = pos0[:, 0] * cos_a - pos0[:, 2] * sin_a
zr = pos0[:, 0] * sin_a + pos0[:, 2] * cos_a

# 散点图。每个粒子的颜色根据其径向距离（hypot）映射到inferno色图上，大小为0.5，
# 透明度为0.85，没有边缘颜色   
scatter = ax.scatter(
    xr, zr * sin_e + pos0[:, 1] * cos_e,
    s=0.5, c=np.hypot(pos0[:, 0], pos0[:, 2]),
    cmap="inferno", norm=Normalize(0, 20), alpha=0.85, edgecolors="none",
)

# 在坐标轴的归一化坐标系中添加文本对象，用于显示当前的模拟步数
step_text = ax.text(
    0.03, 0.97, "", transform=ax.transAxes,
    fontsize=9, color="#8b949e", va="top", fontfamily="monospace",
)


def update(frame):
    pos = trajectory[frame][idx]
    xr = pos[:, 0] * cos_a - pos[:, 2] * sin_a
    zr = pos[:, 0] * sin_a + pos[:, 2] * cos_a
    scatter.set_offsets(np.column_stack([xr, zr * sin_e + pos[:, 1] * cos_e]))
    scatter.set_array(np.hypot(pos[:, 0], pos[:, 2]))
    step_text.set_text(f"step {frame * save_every:>3d}")
    return (scatter, step_text)


anim = FuncAnimation(fig, update, frames=len(trajectory), interval=60, blit=True)
plt.close(fig)

anim.save("galaxy.mp4", writer="ffmpeg", fps=20)

#HTML(anim.to_jshtml())

