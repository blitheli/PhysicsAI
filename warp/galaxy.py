import warp as wp
import numpy as np

num_particles = 35_000
num_steps = 600

central_mass = 5000.0  # Gravitational strength of the central body
softening = 0.5  # Softening length to prevent singularities
dt = 0.005  # Integration time step


def init_galaxy(num_particles, central_mass, disk_radius=20.0, disk_thickness=0.5):
    """Create initial positions and velocities for a spiral galaxy."""
    rng = np.random.default_rng(42)

    # Exponential disk profile for radial distribution
    r = rng.exponential(scale=disk_radius / 3.0, size=num_particles).astype(np.float32)
    r = np.clip(r, 0.1, disk_radius * 2)

    theta = rng.uniform(0, 2 * np.pi, num_particles).astype(np.float32)

    # Add spiral arm perturbation
    theta += 0.4 * np.sin(2 * theta + 0.3 * r)

    # Positions: disk in the XZ plane with thin vertical spread
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    y = rng.normal(0, disk_thickness, num_particles).astype(np.float32) * (
        1.0 - r / (disk_radius * 2)
    )
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    # Keplerian circular velocities: v = sqrt(GM/r), tangent to the orbit
    v_circ = np.sqrt(central_mass / np.maximum(r, 0.1))
    vx = -v_circ * np.sin(theta)
    vz = v_circ * np.cos(theta)
    vy = np.zeros_like(vx)
                
    # Small random dispersion for realism
    dispersion = 0.02
    vx += rng.normal(0, dispersion * v_circ, num_particles)
    vy += rng.normal(0, dispersion * v_circ * 0.1, num_particles)
    vz += rng.normal(0, dispersion * v_circ, num_particles)
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
    velocities[i] = velocity
    positions[i] = positions[i] + velocity * dt


for step in range(num_steps):
    wp.launch(
        integrate_galaxy,
        dim=num_particles,
        inputs=[positions, velocities, central_mass, softening, dt],
    )

print(f"Final positions: {positions}")