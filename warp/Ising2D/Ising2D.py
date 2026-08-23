import numpy as np
import warp as wp

# Check for GPU availability
if wp.get_cuda_device_count() > 0:
    print("✓ GPU detected successfully")
else:
    raise RuntimeError(
        "No CUDA-capable GPU detected. This notebook assumes at least one NVIDIA GPU."
    )

import random
import math


def initialize_lattice(L):
    """
    Initialize a square lattice with random spin orientations.

    Creates an L×L grid where each site contains a magnetic spin that can be
    either +1 (spin up) or -1 (spin down). The initial configuration is
    random.

    Args:
        L (int): Linear size of the lattice (creates L×L grid)

    Returns:
        list[list[int]]: 2-D lattice where lattice[i][j] ∈ {-1, +1}
    """
    lattice = [[random.choice([-1, 1]) for _ in range(L)] for _ in range(L)]
    return lattice


def monte_carlo_step(lattice, L, T):
    """
    Perform one complete Monte Carlo step using the Metropolis-Hastings algorithm

    A Monte Carlo (MC) step consists of attempting to flip each spin in the
    lattice exactly once. We visit sites in random order to avoid
    systematic biases that could arise from sequential scanning.

    For each site, we calculate the energy change ΔE that would result from
    flipping the spin, then accept or reject the flip based on the Boltzmann
    probability

    Physical Parameters (using natural units):
        - J = 1: Interaction strength
        - k_B = 1: Boltzmann constant
        - Periodic boundary conditions

    Args:
        lattice (list[list[int]]): Current spin configuration to update in-place
        L (int): lattice size in terms of number of grid points along one dimension
        T (float): Temperature, T_critical ≈ 2.269 for 2-D Ising model when k_B = J = 1

    Algorithm Details:
        1. Create list of all (i,j) lattice coordinates
        2. Randomly shuffle the update order to avoid artifacts
        3. For each site (i,j):
           a. Calculate sum of 4 nearest neighbors using periodic boundaries
           b. Compute energy change: ΔE = 2 * J * σ_ij * Σ_neighbors
           c. Accept flip with probability P = exp(-ΔE/T)
           d. Update lattice[i][j] *= -1 if flip accepted

    Note:
        The acceptance ratio simplifies to exp(-2 * β * σ_ij * neighbor_sum)
        because we only consider the energy difference, not absolute energy.
    """
    # Convert temperature to inverse temperature (β = 1/T)
    # kB = 1
    beta = 1.0 / T

    # Generate all lattice coordinates and randomize update order
    # This prevents systematic artifacts from sequential scanning
    site_indices = [(r_idx, c_idx) for r_idx in range(L) for c_idx in range(L)]
    random.shuffle(site_indices)

    # Visit each site and attempt to flip its spin according to Metropolis acceptance criterion
    for i, j in site_indices:
        # Current spin at site (i,j)
        spin_ij = lattice[i][j]

        # Calculate sum of 4 nearest neighbors with periodic boundary conditions
        # Modulo arithmetic wraps around lattice edges: (i-1+L)%L and (i+1)%L handle this
        nn_sum = (
            lattice[(i - 1 + L) % L][j]  # neighbor above
            + lattice[(i + 1) % L][j]  # neighbor below
            + lattice[i][(j - 1 + L) % L]  # neighbor left
            + lattice[i][(j + 1) % L]  # neighbor right
        )

        # Energy change from flipping spin: ΔE = -J * (σ_new - σ_old) * Σ_neighbors
        # Since σ_new = -σ_old, we get: ΔE = 2 * J * σ_ij * Σ_neighbors
        # Acceptance probability: P = exp(-β * ΔE) = exp(-β * 2 * σ_ij * Σ_neighbors)
        delta_E = 2 * spin_ij * nn_sum
        acceptance_ratio = math.exp(-beta * delta_E)        

        # Accept flip if random number < acceptance probability
        if delta_E < 0 or random.random() < acceptance_ratio:
            lattice[i][j] *= -1  # Flip the spin
        else:
            pass
        # If flip rejected, spin remains unchanged (no action needed)

def calculate_magnetization(lattice, L):
    """
    Calculate the normalized magnetization of the lattice.

    The magnetization M is the order parameter for the ferromagnetic phase transition.
    It measures the degree of spin alignment in the system.

    Args:
        lattice (list[list[int]]): Current spin configuration
        L (int): Linear lattice size

    Returns:
        float: Normalized magnetization M ∈ [-1, +1]
               M = +1: All spins up (perfect ferromagnetic order)
               M = -1: All spins down (perfect ferromagnetic order)
               M = 0: Equal numbers of up/down spins (disordered)

    Physics Notes:
        - M is the thermal average ⟨Σᵢ σᵢ⟩ / N in equilibrium
        - |M| -> 0 as T -> T_c from below (continuous phase transition)
        - M fluctuates around its equilibrium value due to thermal noise
    """
    total_spin = np.sum(lattice)
    return total_spin / (L * L)

        
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import IPython.display

import os
import time

# --- Simulation Parameters ---
LATTICE_SIZE = 256
TEMPERATURE = 2.269  # Try: T=0.02 (ordered), T=2.269 (critical), T=20.0 (disordered)

print(f"开始模拟 {LATTICE_SIZE}×{LATTICE_SIZE} Ising model at T={TEMPERATURE}")
print(
    f"Critical temperature T_c ≈ 2.269 (this run: {'below' if TEMPERATURE < 2.269 else 'above'} T_c)"
)

# Initialize with random spins (mimics infinite temperature initial condition)
lattice = initialize_lattice(LATTICE_SIZE)

# Set up visualization colormap
# Viridis colormap: dark blue (-1 spins) to bright yellow (+1 spins)
viridis = plt.cm.viridis
norm = Normalize(vmin=-1, vmax=1)

# Collect animation frames
frames = []
print("Running simulation and capturing frames...")

start_time = time.perf_counter()

magnetization_values = []
for step in range(200):  # 200 Monte Carlo steps
    # Evolve the system by one complete lattice sweep
    monte_carlo_step(lattice, LATTICE_SIZE, TEMPERATURE)

    # Calculate and store magnetization
    mag = calculate_magnetization(lattice, LATTICE_SIZE)
    magnetization_values.append(mag)

    # Convert lattice to colored image for visualization
    # Each spin value (-1 or +1) gets mapped to a color
    colored_frame = viridis(norm(np.array(lattice)))

    # Convert to 8-bit RGB for GIF creation
    rgb_frame = (colored_frame[:, :, :3] * 255).astype(np.uint8)
    frames.append(rgb_frame)

    # Progress indicator
    if (step + 1) % 50 == 0:
        print(f"  Step {step + 1}/200 completed")

end_time = time.perf_counter()
print(f"Simulation completed in {end_time - start_time:.2f} seconds")

# Create animated GIF to visualize time evolution
print("Creating animated GIF...")
pil_images = [Image.fromarray(frame) for frame in frames]
output_filename = (
    f"./images/ising-model/python_{LATTICE_SIZE}x{LATTICE_SIZE}_{TEMPERATURE}.gif"
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

IPython.display.Image(output_filename)