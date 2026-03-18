"""Batch peeling experiments with point cloud deformations."""

import argparse
import os

import numpy as np
from diffused_fields.manifold import Pointcloud

from diffused_fields_robotics.utils import BatchPeelingBase
from diffused_fields_robotics.utils.coordinate_utils import (
    compute_body_fixed_coordinate_system,
)
from diffused_fields_robotics.utils.factory import create_primitive_controller


class BatchPeeling(BatchPeelingBase):
    """Specialized batch processor for peeling experiments."""

    def __init__(
        self,
        bend_strength_range=(-3.0, 3.0),
        twist_strength_range=(-3.0, 3.0),
        animate_exp_idx=None,
        animate_sample_idx=0,
        save_gif_path=None,
    ):
        """
        Initialize batch peeling with deformation parameters.

        Args:
            bend_strength_range: (min, max) range for random bending curvature
            twist_strength_range: (min, max) range for random twist strength
            animate_exp_idx: which experiment index to visualize (0-based). If None, no animation.
            animate_sample_idx: which sample index to visualize (default: 0)
            save_gif_path: if not None, export selected experiment as GIF to this path
        """
        super().__init__(
            filename="pear.ply",
            num_experiments=10,  # Use 10 for small file size and fast (50 for paper)
            num_samples=1,  # 1 sample per experiment (each with different random deformations)
            diffusion_scalar=1000,  # fixed diffusion scalar (default)
        )
        self.bend_strength_range = bend_strength_range
        self.twist_strength_range = twist_strength_range
        self.animate_exp_idx = animate_exp_idx
        self.animate_sample_idx = animate_sample_idx
        self.save_gif_path = save_gif_path

    def run_peeling_experiment(self, exp_idx: int, sample_idx: int) -> dict:
        """Run a single peeling experiment with random pointcloud deformations."""
        print(f"\n{'='*80}")
        print(f"Running Peeling Experiment {exp_idx + 1}/{self.num_experiments}")
        print(f"{'='*80}")

        diffusion_scalar = self.diffusion_scalar_arr[exp_idx]

        # Load fresh pointcloud
        pcloud_deformed = Pointcloud(filename=self.filename)

        # Apply random scaling
        scale_factors = np.random.uniform(0.5, 1.5, size=3)
        pcloud_deformed.apply_scaling(scale_factors.tolist())

        # Apply random bending in all axes
        bend_curvature_x = np.random.uniform(*self.bend_strength_range)
        bend_curvature_y = np.random.uniform(*self.bend_strength_range)
        bend_curvature_z = np.random.uniform(*self.bend_strength_range)

        pcloud_deformed.apply_bend(bend_axis=0, curvature=bend_curvature_x)
        pcloud_deformed.apply_bend(bend_axis=1, curvature=bend_curvature_y)
        pcloud_deformed.apply_bend(bend_axis=2, curvature=bend_curvature_z)

        # Apply random twisting in all axes
        twist_strength_x = np.random.uniform(*self.twist_strength_range)
        twist_strength_y = np.random.uniform(*self.twist_strength_range)
        twist_strength_z = np.random.uniform(*self.twist_strength_range)

        pcloud_deformed.apply_twist(axis=0, twist_strength=twist_strength_x)
        pcloud_deformed.apply_twist(axis=1, twist_strength=twist_strength_y)
        pcloud_deformed.apply_twist(axis=2, twist_strength=twist_strength_z)

        # Fix normal orientation and rebuild spatial structures after deformations
        # (they get reset/invalidated during apply_twist/apply_scaling/apply_bend)
        pcloud_deformed.get_normals()  # Re-apply normal_orientation from config
        pcloud_deformed.get_kd_tree()  # Rebuild KD-tree for distance queries

        # Compute body-fixed coordinate system AFTER deformation
        # This ensures the frame is aligned with the deformed source vertices
        body_fixed_frame_R = compute_body_fixed_coordinate_system(
            self.source_vertices, pcloud_deformed.vertices
        )

        # Create controller with deformed pointcloud
        controller = create_primitive_controller(
            primitive_type="peeling",
            pcloud=pcloud_deformed,
            source_vertices=self.source_vertices,
            diffusion_scalar=diffusion_scalar,
        )

        # Run experiment
        controller.run()

        # Optional: visualize or export GIF for selected experiment/sample
        should_animate = (
            self.animate_exp_idx is not None
            and exp_idx == self.animate_exp_idx
            and sample_idx == self.animate_sample_idx
        )
        # If no specific experiment is requested but save_gif_path is set,
        # export GIFs for all experiments with index suffix.
        export_all_gifs = self.animate_exp_idx is None and self.save_gif_path

        if should_animate or export_all_gifs:
            if self.save_gif_path:
                base, ext = os.path.splitext(self.save_gif_path)
                if export_all_gifs:
                    gif_path = f"{base}_{exp_idx}{ext or '.gif'}"
                else:
                    gif_path = self.save_gif_path
                controller.visualize_trajectory(
                    show_tool=True,
                    num_samples=None,
                    save_animation=gif_path,
                )
            else:
                controller.visualize_trajectory(show_tool=True, num_samples=None)

        #  w.r.t. World frame
        trajectory = controller.trajectory
        trajectory_bases = controller.trajectory_local_bases

        # Transform trajectory to body-fixed frame
        body_fixed_trajectory = self._transform_to_body_fixed(
            trajectory, pcloud_deformed.vertices, body_fixed_frame_R
        )

        # Compute velocities in different frames
        if len(trajectory) > 1:
            # World frame velocities
            velocities_world = np.diff(trajectory, axis=0)

            # Local frame (position-varying based on surface/source vertices): uses trajectory_bases[:-1]
            local_bases = trajectory_bases[:-1]
            velocities_local = np.einsum(
                "nij,nj->ni", np.transpose(local_bases, (0, 2, 1)), velocities_world
            )

            # Body-fixed frame (single fixed frame x-axis aligned with source vertices)
            velocities_body_fixed = velocities_world @ body_fixed_frame_R

        else:
            velocities_world = np.zeros((0, 3))
            velocities_local = np.zeros((0, 3))
            velocities_body_fixed = np.zeros((0, 3))

        # Get experiment parameters
        parameters = controller._get_experiment_parameters()

        # Get ground truth transition indices
        transition_indices = getattr(controller, "transition_indices", [])

        return {
            "trajectory": trajectory,
            "trajectory_bases": trajectory_bases,
            "body_fixed_trajectory": body_fixed_trajectory,
            "velocities_world": velocities_world,
            "velocities_local": velocities_local,
            "velocities_body_fixed": velocities_body_fixed,
            "trajectory_length": len(trajectory),
            "body_fixed_frame_R": body_fixed_frame_R,
            # Deformation parameters
            "scale_factors": scale_factors,
            "bend_curvature_x": bend_curvature_x,
            "bend_curvature_y": bend_curvature_y,
            "bend_curvature_z": bend_curvature_z,
            "twist_strength_x": twist_strength_x,
            "twist_strength_y": twist_strength_y,
            "twist_strength_z": twist_strength_z,
            "transition_indices": transition_indices,  # Ground truth transitions
            # Include all data needed for visualization
            "primitive_type": "peeling",
            "object_name": pcloud_deformed.object_name,
            "source_vertices": controller.source_vertices,
            "parameters": parameters,
            "pointcloud": {
                "vertices": pcloud_deformed.vertices,
                "normals": pcloud_deformed.normals,
                "faces": getattr(pcloud_deformed, "faces", None),
                "colors": getattr(pcloud_deformed, "colors", None),
                "local_bases": getattr(pcloud_deformed, "local_bases", None),
            },
        }

    def _transform_to_body_fixed(
        self,
        global_trajectory: np.ndarray,
        pcloud_vertices: np.ndarray,
        body_fixed_frame_R: np.ndarray,
    ) -> np.ndarray:
        """Transform trajectory to body-fixed coordinate system."""
        origin = pcloud_vertices[self.source_vertices[0]]
        body_fixed_trajectory = (global_trajectory - origin) @ body_fixed_frame_R
        return body_fixed_trajectory


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch peeling with random deformations.\n"
            "Optionally visualize or export GIF for a selected experiment."
        )
    )
    parser.add_argument(
        "--animate-exp",
        type=int,
        default=None,
        help="Experiment index (0-based) to visualize with animation.",
    )
    parser.add_argument(
        "--animate-sample",
        type=int,
        default=0,
        help="Sample index (0-based) within the experiment to visualize (default: 0).",
    )
    parser.add_argument(
        "--save-gif",
        metavar="PATH",
        default=None,
        help="If set, export the selected experiment as a GIF to this path.",
    )
    args = parser.parse_args()

    # Create and run batch experiment
    batch_processor = BatchPeeling(
        animate_exp_idx=args.animate_exp,
        animate_sample_idx=args.animate_sample,
        save_gif_path=args.save_gif,
    )

    # Run all experiments with automatic progress tracking and saving
    results = batch_processor.run_experiment_loop(
        batch_processor.run_peeling_experiment,
        save_filename="peeling_batch_results.pkl",
    )

    print(f"✓ Completed {len(results)} experiments")


if __name__ == "__main__":
    main()
