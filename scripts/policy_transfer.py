"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Zero-shot transfer of reaching and distance tracking policy on a 2-D sphere to 
a 3-D point cloud using local reference frames represented using DOF.

"""

import time

import numpy as np
from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import Pointcloud
from diffused_fields.visualization.plotting_ps import *

# Import stable_baselines3 AFTER diffused_fields
from stable_baselines3 import PPO

# Import config from this package
from diffused_fields_robotics.core.config import get_policy_dir

# Configuration
# ==============================================================================
# Select the object point cloud
filename = "spot.ply"

# RL model path
model_path = get_policy_dir() / "ppo_pointmass_circle_local.zip"

# Simulation parameters
max_steps = 300
distance_to_surface = -0.005  # Target distance from surface
step_size_scale = 0.1  # Scale for adaptive step size
n_trajectories = 20  # Number of different initial positions to sample

# Precomputations
# ==============================================================================
print("Loading point cloud and setting up diffusion...")
pcloud = Pointcloud(filename=filename)

# Select random target vertex on the point cloud
random_vertex = np.random.randint(0, len(pcloud.vertices))
# random_vertex = 3394  # for paper
source_vertices = np.array([random_vertex])

# Setup scalar diffusion for the point cloud
scalar_diffusion = PointcloudScalarDiffusion(pcloud)
scalar_diffusion.source_vertices = source_vertices
scalar_diffusion.get_local_bases()

# Setup Walk-on-Spheres diffusion for the ambient space
boundaries = [pcloud]
wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=pcloud.get_mean_edge_length() * 2,
)

# Load the trained RL model
print(f"Loading RL model from {model_path}...")
model = PPO.load(str(model_path))
print("Model loaded successfully!")


# RL Trajectory Rollout for Multiple Initial Positions
# ==============================================================================
print(
    f"\nStarting RL trajectory rollout for {n_trajectories} different initial positions..."
)

all_trajectories = []
start_time = time.time()

for traj_idx in range(n_trajectories):
    print(f"\n=== Trajectory {traj_idx + 1}/{n_trajectories} ===")

    # Initialize random agent position by sampling from point cloud
    random_start_vertex = np.random.randint(0, len(pcloud.vertices))
    random_distance = np.random.uniform(0, 0.05)
    agent_pos = (
        pcloud.vertices[random_start_vertex]
        - pcloud.normals[random_start_vertex] * random_distance
    )

    x_current = agent_pos.copy()
    trajectory = [x_current.copy()]

    for step in range(max_steps):
        # Compute observation: distances to object and target
        distance_to_object, closest_idx = pcloud.get_closest_points(x_current)
        distance_to_target = np.linalg.norm(
            pcloud.vertices[source_vertices[0]] - x_current
        )

        # Construct observation: [distance_to_object, distance_to_target]
        obs = np.array([distance_to_object, distance_to_target], dtype=np.float32)

        # Get action from RL model
        action, _states = model.predict(obs, deterministic=True)

        # Simulate environment step using diffusion framework
        batch_points = wos_diffusion.get_batch_from_point(x_current)
        local_basis, _, _ = wos_diffusion.diffuse_rotations(batch_points)

        # Apply action in local coordinates
        velocity_local = np.array([action[1], 0.0, action[0]])
        velocity_world = local_basis @ velocity_local * 0.001

        # Update position
        x_next = x_current + velocity_world
        trajectory.append(x_next.copy())

        # Check termination conditions
        if distance_to_target < 0.012:
            print(f"  Reached target at step {step}")
            break

        x_current = x_next

    trajectory = np.asarray(trajectory)
    all_trajectories.append(trajectory)
    print(f"  Total steps: {len(trajectory)}")

print(f"\nAll rollouts finished in {time.time() - start_time:.2f} seconds")

# Visualization
# ==============================================================================
print("\nVisualizing trajectories...")
import polyscope as ps

ps.init()
set_camera_and_plane()
plot_orientation_field(vertices=pcloud.vertices, name="pcloud")

# Generate different colors for each trajectory
import matplotlib.pyplot as plt

cmap = plt.cm.get_cmap("tab10")
colors = [cmap(i / n_trajectories)[:3] for i in range(n_trajectories)]

# Plot all trajectories as curve networks and their start points
for traj_idx, trajectory in enumerate(all_trajectories):
    # Plot trajectory curve
    ps.register_curve_network(
        f"trajectory_{traj_idx}",
        trajectory,
        edges="line",
        color=colors[traj_idx],
        radius=0.005,
    )

    # Plot start point with same color as trajectory
    start_point = trajectory[0:1]  # Keep as 2D array (1, 3)
    ps.register_point_cloud(
        f"start_{traj_idx}", start_point, radius=0.012, color=colors[traj_idx]
    )

# Plot target point as a larger blue point
target_point = pcloud.vertices[scalar_diffusion.source_vertices]
ps.register_point_cloud("target", target_point, radius=0.015, color=[0.0, 0.0, 1.0])

print("Opening visualization window...")
ps.show()
