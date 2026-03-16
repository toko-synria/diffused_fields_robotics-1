"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import copy
import os
import pickle
from datetime import datetime

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from diffused_fields import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.visualization.plotting_ps import *


class pcloudActionPrimitives(object):
    """
    General class for other primitives to inherit from
    """

    def __init__(self, pcloud, primitive_type, **kwargs):
        self.pcloud = pcloud
        self.primitive_type = primitive_type

        # Load parameters from config
        self.load_parameters()

        # Set default step_size if not already set by config
        if not hasattr(self, "step_size"):
            self.step_size = 0.001

        # Set common attributes from kwargs (only if provided)
        if "start_vertex" in kwargs:
            self.start_vertex = kwargs["start_vertex"]
        if "end_vertex" in kwargs:
            self.end_vertex = kwargs["end_vertex"]
        if "source_vertices" in kwargs:
            self.source_vertices = kwargs["source_vertices"]
        diffusion_scalar = kwargs.get("diffusion_scalar", None)

        # Allow derived classes to set additional attributes
        self._setup_primitive_specific_attributes(**kwargs)

        # Setup diffusion systems
        self._initialize_diffusion_systems(diffusion_scalar)

        # Post-initialization setup (after diffusion systems are ready)
        self._post_initialization_setup()

        # Initialize trajectory
        self.init_trajectory()

    def _setup_primitive_specific_attributes(self, **kwargs):
        """Override in derived classes for primitive-specific setup"""
        pass

    def _post_initialization_setup(self):
        """Override in derived classes for setup that needs initialized diffusion systems"""
        pass

    def _initialize_diffusion_systems(self, diffusion_scalar):
        """Initialize scalar diffusion and walk-on-spheres systems"""
        if diffusion_scalar is None:
            diffusion_scalar = self.diffusion_scalar

        self.scalar_diffusion = PointcloudScalarDiffusion(
            self.pcloud, diffusion_scalar=diffusion_scalar
        )

        # Setup source vertices
        if hasattr(self, "source_vertices") and self.source_vertices is not None:
            if type(self.source_vertices) == str:
                self.scalar_diffusion.get_endpoints()
                self.scalar_diffusion.source_vertices = self.scalar_diffusion.endpoints
            else:
                self.scalar_diffusion.source_vertices = self.source_vertices
        else:
            # Will use the default source_vertices from loaded parameters
            self.scalar_diffusion.source_vertices = self.source_vertices

        self.scalar_diffusion.get_local_bases()

        # Setup Walk-on-Spheres diffusion for ambient space
        boundaries = [self.pcloud]
        self.wos = WalkOnSpheresDiffusion(
            boundaries=boundaries,
            convergence_threshold=self.pcloud.get_mean_edge_length() * 2,
        )

    def init_trajectory(self):
        """Initialize trajectory with optional safety offset from starting point.

        The safety offset moves num_init_steps away from the initial vertex in the
        local x-direction (longitudinal/radial) before beginning the main trajectory.
        This helps avoid boundary/source vertex issues.
        """
        self.trajectory_local_bases = []
        self.x_arr = []
        self.actions = []

        # Determine starting point
        x0 = self._get_initial_point()

        # Get number of initialization steps (default 0 for backwards compatibility)
        num_init_steps = getattr(self, "num_init_steps", 0)

        if num_init_steps > 0:
            # Move num_init_steps in x-direction (longitudinal/radial) for safety
            # We do this without appending to trajectory - just to find safe starting position
            print(f"Moving {num_init_steps} steps from starting vertex for safety...")
            x_current = x0
            for _ in range(num_init_steps):
                x_current, local_basis = self.local_step(
                    x_current, direction=0, sign=1  # x-direction (longitudinal/radial)
                )
                x_current, _, _ = self.pcloud.correct_distance_smooth(
                    x_current, self.distance_to_surface
                )

            # Use the offset position as our actual starting point
            x0 = x_current
            print(f"Safety offset complete. Starting from interior position.")

        # Set the starting point (either original or after safety offset)
        self.x_arr.append(x0)
        # Get batch points for diffusion
        batch_points = self.wos.get_batch_from_point(x0)
        # Compute local bases using diffusion
        local_basis, _, _ = self.wos.diffuse_rotations(batch_points)
        self.trajectory_local_bases.append(local_basis)

    def _get_initial_point(self):
        """Get the initial starting point for trajectory. Override for custom behavior."""
        if hasattr(self, "start_vertex"):
            return (
                self.pcloud.vertices[self.start_vertex]
                + self.pcloud.normals[self.start_vertex] * self.distance_to_surface
            )
        else:
            return (
                self.pcloud.vertices[self.source_vertices[0]]
                + self.pcloud.normals[self.source_vertices[0]]
                * self.distance_to_surface
            )

    def load_parameters(self):
        """Load parameters using the new hierarchical config system."""
        from ..core.config import get_action_primitive_config

        # Load merged configuration (defaults + object-specific overrides)
        merged_config = get_action_primitive_config(
            self.primitive_type, self.pcloud.object_name
        )
        # print(f"Loaded parameters for {self.primitive_type}: {merged_config}")

        # Set all parameters as attributes
        self._set_parameters(merged_config)

    def _set_parameters(self, parameters: dict):
        """Recursively set parameters as attributes."""

        def set_attributes(obj, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    # Create a new attribute (or use existing) and recurse
                    sub_obj = getattr(obj, key, type("SubParams", (), {})())
                    set_attributes(sub_obj, value)
                    setattr(obj, key, sub_obj)
                else:
                    setattr(obj, key, value)
                    # print(f"Set {key} as {value}")

        set_attributes(self, parameters)

    def local_step(self, x, direction, sign):
        # Get batch points for diffusion
        batch_points = self.wos.get_batch_from_point(x)

        # Compute local bases using diffusion
        local_basis, _, _ = self.wos.diffuse_rotations(batch_points)

        # Update the next position
        next_x = x + (local_basis[:, direction] * self.step_size * sign)
        return next_x, local_basis

    def check_endpoint_reached(self, x_next, local_basis=None):
        """Check if endpoint is reached within tolerance, accounting for surface distance."""
        if not hasattr(self, "end_point"):
            return False

        tolerance = getattr(self, "endpoint_tolerance_multiplier", 30.0)
        dist2end = np.linalg.norm(x_next - self.end_point)

        reached = (
            np.abs(dist2end - np.abs(self.distance_to_surface))
            < self.step_size * tolerance
        )

        if reached:
            self.reached_end_point = True

        return reached

    def move_multistep(
        self,
        num_steps,
        x0,
        direction,
        sign,
        project=False,
        distance_to_surface=0.0,
        terminal_condition=None,
    ):
        x_next = x0
        for _ in range(num_steps):
            if type(direction) == list:
                for dir, sgn in zip(direction, sign):
                    x_next, local_basis = self.local_step(x_next, dir, sgn)

                    self.x_arr.append(x_next)
                    self.trajectory_local_bases.append(local_basis)
                    print(f"Step {_ + 1} of {num_steps}: {x_next}")
            else:
                x_next, local_basis = self.local_step(x_next, direction, sign)
            if project:
                x_next, _, _ = self.pcloud.correct_distance_smooth(
                    x_next, distance_to_surface
                )
            if terminal_condition is not None:
                if terminal_condition(x_next, local_basis):
                    break
            # print(f"Step {_ + 1} of {num_steps}: {x_next}")
            self.x_arr.append(x_next)
            self.trajectory_local_bases.append(local_basis)

    def _tool_vertices_at_pose(self, mesh, position, orientation):
        """Compute tool mesh vertices at a single pose (for animation export)."""
        mesh_c = copy.deepcopy(mesh.mesh)
        target_pos = position + mesh.center_offset
        mesh_c.translate(target_pos, relative=False)
        rot = (
            orientation
            @ R.from_euler("xyz", [0, 0, 0], degrees=False).as_matrix()
        )
        mesh_c.rotate(rot, center=mesh_c.get_center() - mesh.center_offset)
        return np.asarray(mesh_c.vertices)

    def visualize_trajectory(self, show_tool=False, num_samples=None, save_animation=None):
        # Store visualization parameters for later use
        self.visualization_num_samples = num_samples

        # Visualize the diffusion on the point cloud
        # ==============================================================================
        vector_length = 0.05 / 2
        vector_radius = 0.035 / 2
        point_radius = 0.003
        curve_radius = 0.003

        ps.init()

        ps.register_curve_network(
            f"Trajectory",
            self.trajectory,
            edges="line",
            radius=curve_radius,
            transparency=1.0,
            color=[0, 0, 0],
        )

        # Point Cloud
        # ==============================================================================
        ps_cloud = plot_orientation_field(
            self.pcloud.vertices,
            self.pcloud.local_bases,
            name="object point cloud",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_vector=True,
            enable_z=True,
            color=self.pcloud.colors,
        )
        # Add colors manually since the external plot_orientation_field ignores them
        if hasattr(self.pcloud, "colors") and self.pcloud.colors is not None:
            ps_cloud.add_color_quantity("colors", self.pcloud.colors, enabled=True)
        ps_cloud.add_vector_quantity("z", self.pcloud.normals)

        plot_orientation_field(
            self.trajectory,
            self.trajectory_local_bases,
            name="trajectory frames",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_x=False,
        )

        # Keypoints
        # ==============================================================================
        if len(self.source_vertices) == 1:

            ps_keypoints = plot_orientation_field(
                self.pcloud.vertices[self.source_vertices],
                # self.pcloud.local_bases[self.pcloud.source_vertices],
                name="keypoints",
                vector_length=vector_length,
                vector_radius=vector_radius,
            )
            # Add blue color manually since plot_orientation_field ignores color parameter
            blue_color = np.array([[0, 0, 1]])
            ps_keypoints.add_color_quantity("colors", blue_color, enabled=True)
        else:
            ps_keypoints = plot_orientation_field(
                self.pcloud.vertices[self.source_vertices],
                # self.pcloud.local_bases[self.pcloud.source_vertices],
                name="keypoints",
                point_radius=0.0326,
                vector_length=vector_length,
                vector_radius=vector_radius,
            )
            # Add blue color manually since plot_orientation_field ignores color parameter
            blue_colors = np.tile([0, 0, 1], (len(self.source_vertices), 1))
            ps_keypoints.add_color_quantity("colors", blue_colors, enabled=True)

        if show_tool:
            tool_mesh = import_tool_mesh(self.tool)
            if num_samples is None or save_animation is not None:
                indices = np.linspace(
                    0, len(self.trajectory) - 1, len(self.trajectory) - 1, dtype=int
                )
                if save_animation is not None:
                    max_frames = 80
                    indices = np.linspace(
                        0, len(self.trajectory) - 1, min(max_frames, len(indices)), dtype=int
                    )
            else:
                # Create evenly spaced indices but exclude first and last
                # Add 2 to num_samples to account for endpoints we'll exclude
                all_indices = np.linspace(
                    0, len(self.trajectory) - 1, num_samples + 2, dtype=int
                )
                # Remove first and last indices
                indices = all_indices[1:-1]
            downsampled_trajectory = self.trajectory[indices]
            downsampled_trajectory_local_bases = self.trajectory_local_bases[indices]

            if save_animation is not None:
                # Export animation to GIF (no window show)
                orients = -downsampled_trajectory_local_bases
                verts0 = self._tool_vertices_at_pose(
                    tool_mesh,
                    downsampled_trajectory[0],
                    orients[0],
                )
                faces = np.asarray(tool_mesh.mesh.triangles)
                ps_tool = ps.register_surface_mesh(
                    "moving mesh",
                    verts0,
                    faces,
                    transparency=1.0,
                    color=[125, 125, 125],
                )
                # Set camera: look at scene center from side-above angle
                scene_center = np.mean(
                    np.vstack((self.trajectory, self.pcloud.vertices)), axis=0
                )
                scene_scale = 0.8
                camera_offset = np.array([0.2, -0.1, -0.3]) * scene_scale
                ps.look_at(
                    (scene_center + camera_offset).tolist(),
                    scene_center.tolist(),
                )
                frames = []
                for i in range(len(downsampled_trajectory)):
                    v = self._tool_vertices_at_pose(
                        tool_mesh,
                        downsampled_trajectory[i],
                        orients[i],
                    )
                    ps_tool.update_vertex_positions(v)
                    img = ps.screenshot_to_buffer(transparent_bg=False, vertical_flip=True)
                    frames.append(img)
                try:
                    import imageio
                    # imageio expects (H,W,C); screenshot is (H,W,4) or (H,W,3)
                    duration = 0.08
                    imageio.mimwrite(save_animation, frames, duration=duration, loop=0)
                    print(f"Animation saved to {save_animation}")
                except ImportError:
                    print("Install imageio to export GIF: pip install imageio")
                return
            if num_samples is None:
                animate_tool_trajectory(
                    downsampled_trajectory,
                    -downsampled_trajectory_local_bases,
                    tool_mesh,
                )
            else:
                plot_orientation_field(
                    downsampled_trajectory,
                    -downsampled_trajectory_local_bases,
                    name="local frames",
                    vector_length=vector_length,
                    vector_radius=vector_radius,
                    point_radius=point_radius,
                    enable_vector=True,
                )

                plot_tool_trajectory(
                    downsampled_trajectory,
                    -downsampled_trajectory_local_bases,
                    tool_mesh,
                )

        ps.show()

    def save_results(self, filepath=None, include_pointcloud=True):
        """
        Save experiment results to a pickle file.

        Args:
            filepath (str, optional): Path to save the results. If None, generates automatic filename.
            include_pointcloud (bool): Whether to include the full pointcloud data.

        Returns:
            str: Path to the saved file
        """
        if not hasattr(self, "trajectory") or self.trajectory is None:
            raise RuntimeError("No trajectory data found. Run the experiment first.")

        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{self.primitive_type}_{self.pcloud.object_name}_{timestamp}.pkl"
            )
            filepath = os.path.join("results", filename)

        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Prepare data to save
        results_data = {
            "primitive_type": self.primitive_type,
            "object_name": self.pcloud.object_name,
            "trajectory": self.trajectory,
            "trajectory_local_bases": self.trajectory_local_bases,
            "source_vertices": getattr(self, "source_vertices", None),
            "parameters": self._get_experiment_parameters(),
            "timestamp": datetime.now().isoformat(),
        }

        # Add primitive-specific data
        if hasattr(self, "end_point"):
            results_data["end_point"] = self.end_point
        if hasattr(self, "reached_end_point"):
            results_data["reached_end_point"] = self.reached_end_point

        # Optionally include pointcloud data
        if include_pointcloud:
            results_data["pointcloud"] = {
                "vertices": self.pcloud.vertices,
                "normals": self.pcloud.normals,
                "faces": getattr(self.pcloud, "faces", None),
                "colors": getattr(self.pcloud, "colors", None),
                "local_bases": getattr(self.pcloud, "local_bases", None),
            }
        else:
            results_data["pointcloud_filename"] = self.pcloud.filename

        # Save to pickle file
        with open(filepath, "wb") as f:
            pickle.dump(results_data, f)

        print(f"Results saved to: {filepath}")
        return filepath

    @classmethod
    def load_results(cls, filepath):
        """
        Load experiment results from a pickle file.

        Args:
            filepath (str): Path to the saved results file

        Returns:
            dict: Loaded experiment data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, "rb") as f:
            results_data = pickle.load(f)

        print(f"Results loaded from: {filepath}")
        print(
            f"Experiment: {results_data['primitive_type']} on {results_data['object_name']}"
        )
        print(f"Trajectory points: {len(results_data['trajectory'])}")

        return results_data

    def _convert_to_dict(self, obj):
        """
        Convert an object (including dynamic SubParams) to a dictionary for pickle serialization.

        Args:
            obj: Object to convert

        Returns:
            dict or original object if not convertible
        """
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                result[key] = self._convert_to_dict(value)  # Recursive conversion
            return result
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_dict(item) for item in obj]
        else:
            return obj

    def _get_experiment_parameters(self):
        """
        Extract experiment parameters for saving.

        Returns:
            dict: Dictionary of experiment parameters
        """
        params = {}

        # Common parameters
        param_names = [
            "diffusion_scalar",
            "step_size",
            "distance_to_surface",
            "num_init_steps",
            "num_slices",
            "num_slicing_steps",
            "num_slide_steps",
            "num_peels",
            "num_peeling_steps",
            "num_loops",
            "num_tangential_steps",
            "num_radial_steps",
            "num_cut_steps",
            "tool",
            "visualization_num_samples",
        ]

        for param in param_names:
            if hasattr(self, param):
                param_value = getattr(self, param)
                params[param] = self._convert_to_dict(param_value)

        return params

    @staticmethod
    def visualize_from_results(results_data, show_tool=False, num_samples=None):
        """
        Visualize experiment results loaded from a saved file.

        Args:
            results_data (dict): Loaded results data from load_results()
            show_tool (bool): Whether to show the tool visualization
            num_samples (int, optional): Number of trajectory samples to show.
                                       If None, uses the stored value from original experiment.
        """
        # Use stored num_samples if not provided
        if num_samples is None and "parameters" in results_data:
            num_samples = results_data["parameters"].get(
                "visualization_num_samples", None
            )
        if "pointcloud" not in results_data:
            raise ValueError(
                "Pointcloud data not found in results. Re-save with include_pointcloud=True"
            )

        trajectory = results_data["trajectory"]
        trajectory_local_bases = results_data["trajectory_local_bases"]
        pcloud_data = results_data["pointcloud"]

        # Visualization parameters
        vector_length = 0.05 / 2
        vector_radius = 0.035 / 2
        point_radius = 0.003
        curve_radius = 0.003

        ps.init()

        # Trajectory curve
        ps.register_curve_network(
            f"Trajectory",
            trajectory,
            edges="line",
            radius=curve_radius,
            transparency=1.0,
            color=[0, 0, 0],
        )

        # Point cloud
        ps_cloud = plot_orientation_field(
            pcloud_data["vertices"],
            pcloud_data["local_bases"],
            name="object point cloud",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_vector=False,
            enable_x=False,
            color=pcloud_data["colors"],
        )

        # Add colors and normals
        if pcloud_data["colors"] is not None:
            ps_cloud.add_color_quantity("colors", pcloud_data["colors"], enabled=True)
        if pcloud_data["normals"] is not None:
            ps_cloud.add_vector_quantity("z", pcloud_data["normals"])

        # Trajectory frames
        plot_orientation_field(
            trajectory,
            trajectory_local_bases,
            name="trajectory frames",
            vector_length=vector_length,
            vector_radius=vector_radius,
            point_radius=point_radius,
            enable_vector=False,
            enable_x=True,
        )

        # Source points
        if (
            "source_vertices" in results_data
            and results_data["source_vertices"] is not None
        ):
            source_vertices = results_data["source_vertices"]
            if len(source_vertices) == 1:
                plot_orientation_field(
                    pcloud_data["vertices"][source_vertices],
                    name="source",
                    vector_length=vector_length,
                    vector_radius=vector_radius,
                )
            else:
                plot_orientation_field(
                    pcloud_data["vertices"][source_vertices],
                    name="source",
                    point_radius=0.0326,
                    vector_length=vector_length,
                    vector_radius=vector_radius,
                )

        # Tool visualization (if requested)
        if show_tool:
            tool_data = None
            if "parameters" in results_data and "tool" in results_data["parameters"]:
                tool_data = results_data["parameters"]["tool"]
            elif "tool" in results_data:
                tool_data = results_data["tool"]

            if tool_data is None:
                print(
                    "Warning: Tool information not found in results. Re-save results to include tool data."
                )
            else:
                try:
                    # Convert dictionary back to object-like structure if needed
                    if isinstance(tool_data, dict):

                        class ToolData:
                            def __init__(self, data_dict):
                                for key, value in data_dict.items():
                                    setattr(self, key, value)

                        tool_obj = ToolData(tool_data)
                    else:
                        tool_obj = tool_data

                    tool_mesh = import_tool_mesh(tool_obj)

                    if num_samples is None:
                        indices = np.linspace(
                            0, len(trajectory) - 1, len(trajectory) - 1, dtype=int
                        )
                    else:
                        indices = np.linspace(
                            0, len(trajectory) - 1, num_samples, dtype=int
                        )
                    downsampled_trajectory = trajectory[indices]
                    downsampled_trajectory_local_bases = trajectory_local_bases[indices]

                    if num_samples is None:
                        animate_tool_trajectory(
                            downsampled_trajectory,
                            -downsampled_trajectory_local_bases,
                            tool_mesh,
                        )
                    else:
                        plot_orientation_field(
                            downsampled_trajectory,
                            -downsampled_trajectory_local_bases,
                            name="local frames",
                            vector_length=vector_length,
                            vector_radius=vector_radius,
                            point_radius=point_radius,
                            enable_vector=True,
                        )

                        plot_tool_trajectory(
                            downsampled_trajectory,
                            -downsampled_trajectory_local_bases,
                            tool_mesh,
                        )
                except Exception as e:
                    print(f"Warning: Could not load or visualize tool: {e}")

        ps.show()


class Cutting(pcloudActionPrimitives):

    def __init__(self, pcloud, primitive_type="cutting", **kwargs):
        super().__init__(pcloud, primitive_type, **kwargs)

    def _post_initialization_setup(self):
        """Setup cutting-specific attributes after diffusion is initialized"""
        end_vertex = self.scalar_diffusion.source_vertices[1]
        self.end_point = self.pcloud.vertices[end_vertex]
        self.endpoint_tolerance_multiplier = 20.0

    def run(self):
        self.move_multistep(
            self.num_cut_steps,
            self.x_arr[-1],
            direction=0,
            sign=1,
            project=True,
            terminal_condition=self.check_endpoint_reached,
        )

        self.trajectory = np.array(self.x_arr)
        self.trajectory_local_bases = np.array(self.trajectory_local_bases)


class Slicing(pcloudActionPrimitives):
    def __init__(self, pcloud, primitive_type="slicing", **kwargs):
        super().__init__(pcloud, primitive_type, **kwargs)

        # Use a large number so it will terminate via check_endpoint_reached
        if not hasattr(self, "num_slices"):
            self.num_slices = 30

    def _post_initialization_setup(self):
        """Setup slicing-specific attributes after diffusion is initialized"""
        end_vertex = self.scalar_diffusion.source_vertices[1]
        self.end_point = self.pcloud.vertices[end_vertex]
        self.reached_end_point = False
        self.endpoint_tolerance_multiplier = 20.0  # Higher value = earlier termination

    def run(self):
        for slice_idx in range(self.num_slices):
            print(f"Performing slice {slice_idx + 1} of {self.num_slices}")

            # Move down to perform the slice
            self.move_multistep(
                self.num_slicing_steps,
                self.x_arr[-1],
                direction=2,  # towards inside
                sign=1,
                project=False,
            )

            # Move back up to original surface level
            self.move_multistep(
                self.num_slicing_steps,
                self.x_arr[-1],
                direction=2,  # towards inside
                sign=-1,
                project=False,
            )

            # Move to the next slice
            self.move_multistep(
                self.num_slide_steps,
                self.x_arr[-1],
                direction=0,  # longitudinal axis
                sign=1,
                project=True,
                distance_to_surface=self.distance_to_surface,
                terminal_condition=self.check_endpoint_reached,
            )

            # Check if endpoint was reached during the slide
            if self.reached_end_point:
                print(
                    f"Endpoint reached after upward movement in slice {slice_idx + 1}. Stopping slicing."
                )
                print(f"Slicing completed after {slice_idx + 1} slices")
                break

        self.trajectory = np.array(self.x_arr)
        self.trajectory_local_bases = np.array(self.trajectory_local_bases)


class Coverage(pcloudActionPrimitives):
    def __init__(self, pcloud, primitive_type="coverage", **kwargs):
        super().__init__(pcloud, primitive_type, **kwargs)

        # Set default loop_distance_threshold if not provided by config
        if not hasattr(self, "loop_distance_threshold"):
            self.loop_distance_threshold = 0.01

    def _get_initial_point(self):
        """Override to support boundary detection for coverage."""
        if hasattr(self, "start_vertex"):
            return (
                self.pcloud.vertices[self.start_vertex]
                + self.pcloud.normals[self.start_vertex] * self.distance_to_surface
            )
        else:
            # Get boundary points and select a random one
            self.pcloud.get_boundary()
            boundary_vertices = np.where(self.pcloud.is_boundary_arr)[0]
            # Use random selection for more robust starting point
            start_vertex = np.random.choice(boundary_vertices)
            start_vertex = boundary_vertices[0]
            return (
                self.pcloud.vertices[start_vertex]
                + self.pcloud.normals[start_vertex] * self.distance_to_surface
            )

    def visualize_trajectory(self, show_tool=False, num_samples=None):
        """Override to visualize all boundary points as keypoints"""
        # Get boundary points
        if not hasattr(self.pcloud, "is_boundary_arr"):
            self.pcloud.get_boundary()
        boundary_vertices = np.where(self.pcloud.is_boundary_arr)[0]

        # Store boundary vertices as source_vertices for visualization
        original_source_vertices = self.source_vertices
        self.source_vertices = boundary_vertices

        # Call parent visualization
        super().visualize_trajectory(show_tool=show_tool, num_samples=num_samples)

        # Restore original source_vertices
        self.source_vertices = original_source_vertices

    def check_terminal_condition(self, x_next, local_basis):
        """
        Check if a tangential loop is complete using Euclidean distance.
        """
        # Initialize step counter if not present
        if not hasattr(self, "_loop_step_count"):
            self._loop_step_count = 0
        self._loop_step_count += 1

        # Safety: Force completion after too many steps (prevent infinite loops)
        if self._loop_step_count > 1000:
            print(
                f"Loop force-completed after {self._loop_step_count} steps (safety limit)"
            )
            self._loop_step_count = 0
            return True

        # Use Euclidean distance from loop origin
        euclidean_distance = np.linalg.norm(x_next - self.x_origin)

        # Check if the loop should terminate based on distance
        # Loop is complete when we return close to origin after moving away
        if euclidean_distance - self.euclidean_distance_prev < 0:
            self.triggered = True
        if self.triggered and euclidean_distance < self.loop_distance_threshold:
            print(f"Completed a loop (in {self._loop_step_count} steps)")
            self._loop_step_count = 0  # Reset for next loop
            return True
        self.euclidean_distance_prev = euclidean_distance
        return False

    def check_coverage_complete(self):
        """
        Check if coverage is complete by measuring the circumference of the current loop.
        When the loop becomes very small (near zero circumference), we've reached the center.
        Uses multiple criteria for robustness.
        """
        # Need at least one completed loop to measure
        if not hasattr(self, "loop_path_lengths") or len(self.loop_path_lengths) < 1:
            return False

        current_loop_length = self.loop_path_lengths[-1]

        # Criterion 1: Absolute minimum loop circumference threshold
        # Use a very small threshold relative to the first loop to avoid premature termination
        # Only check after we have at least 2 loops to establish baseline
        if len(self.loop_path_lengths) >= 2:
            first_loop_length = self.loop_path_lengths[0]
            # Terminate only when current loop is less than 5% of the first loop
            min_loop_threshold = max(first_loop_length * 0.05, self.step_size * 5)

            if current_loop_length < min_loop_threshold:
                print(
                    f"Coverage complete: loop circumference too small ({current_loop_length:.6f} < {min_loop_threshold:.6f}, {current_loop_length/first_loop_length*100:.1f}% of first loop)"
                )
                return True

        # Criterion 2: Check if loop size is decreasing too slowly (potential infinite loop)
        if len(self.loop_path_lengths) >= 5:
            # Need at least 5 loops before checking stall condition
            # Compare last 4 loops to see if we're making progress
            recent_loops = self.loop_path_lengths[-4:]
            # Check if loops are not shrinking (within 3% tolerance)
            if all(
                abs(recent_loops[i] - recent_loops[i + 1]) / recent_loops[i] < 0.03
                for i in range(len(recent_loops) - 1)
            ):
                print(
                    f"Coverage complete: loop size not decreasing (last 4 loops: {[f'{l:.6f}' for l in recent_loops]})"
                )
                return True

        # Criterion 3: Check if we've done too many loops (safety)
        max_loops = getattr(self, "num_loops", 20)
        if self.loop_count >= max_loops:
            print(f"Coverage complete: reached maximum number of loops ({max_loops})")
            return True

        return False

    def run(self):
        sign_y = 1  # tangential_direction

        # Track loop path lengths to detect when loops become too small
        self.loop_path_lengths = []
        self.loop_count = 0

        # The actual termination will be based on automatic loop size detection
        max_loops = getattr(self, "num_loops", 30)

        for loop_idx in range(max_loops):
            self.loop_count = loop_idx + 1

            # Store this variables to check tangential loop completion
            self.x_origin = self.x_arr[-1]  # Starting point of the current loop
            loop_start_idx = len(self.x_arr) - 1  # Track where this loop starts

            # Get reference local frame at loop start for angular tracking
            # Get batch points for diffusion to get local basis
            batch_points = self.wos.get_batch_from_point(self.x_origin)
            reference_basis, _, _ = self.wos.diffuse_rotations(batch_points)

            # Store reference directions for angular comparison
            self.reference_x = reference_basis[:, 0]  # Reference radial direction
            self.reference_z = reference_basis[:, 2]  # Reference normal direction

            # Initialize angular tracking variables
            self.total_angular_change = 0.0
            self.prev_signed_angle = None

            # Initialize Euclidean distance tracking
            self.euclidean_distance_prev = 0
            self.triggered = False

            # STEP 1: Complete tangential loop
            # Use Euclidean distance-based terminal condition
            self.move_multistep(
                1000,  # large number to ensure completion but will be cut by terminal condition
                self.x_arr[-1],
                direction=1,  # tangential direction
                sign=sign_y,  # flip directions
                distance_to_surface=self.distance_to_surface,
                project=True,
                terminal_condition=self.check_terminal_condition,
            )
            # Flip sign after completing tangential direction
            sign_y *= -1

            # Calculate the path length of this loop (tangential movement)
            loop_end_idx = len(self.x_arr) - 1
            loop_path_length = 0.0
            for i in range(loop_start_idx, loop_end_idx):
                loop_path_length += np.linalg.norm(self.x_arr[i + 1] - self.x_arr[i])
            self.loop_path_lengths.append(loop_path_length)

            print(f"Loop {self.loop_count}: circumference = {loop_path_length:.6f}")

            # Check if coverage is complete (after completing tangential loop)
            if self.check_coverage_complete():
                break

            # STEP 2: Move inward in radial direction for next loop
            self.move_multistep(
                self.num_radial_steps,
                self.x_arr[-1],
                direction=0,  # radial direction
                sign=1,
                distance_to_surface=self.distance_to_surface,
                project=True,
            )

        print(f"Coverage completed after {self.loop_count} loops")
        self.trajectory = np.array(self.x_arr)
        self.trajectory_local_bases = np.array(self.trajectory_local_bases)


class Peeling(pcloudActionPrimitives):
    def __init__(self, pcloud, primitive_type="peeling", **kwargs):
        super().__init__(pcloud, primitive_type, **kwargs)

    def _post_initialization_setup(self):
        """Setup peeling-specific attributes after diffusion is initialized"""
        self.end_point = self.pcloud.vertices[self.source_vertices[1]]
        self.force_list = []
        self.endpoint_tolerance_multiplier = 10.0

    def run(self):
        self.x_home = np.copy(self.x_arr[0])

        # Track ground truth transition indices
        self.transition_indices = []

        for _ in range(self.num_peels):
            print(f"Performing peel {_ + 1} of {self.num_peels}")
            # Peeling movement in longitudinal direction
            self.move_multistep(
                500,  # large number to ensure completion but will be cut by terminal condition
                self.x_arr[-1],
                direction=0,  # longitudinal direction
                sign=1,
                project=True,
                distance_to_surface=self.distance_to_surface,
                terminal_condition=self.check_endpoint_reached,
            )
            self.transition_indices.append(len(self.x_arr) - 1)

            # Lift the tool away from the surface
            lift_steps = int(-self.retract_distance_to_surface / self.step_size)
            self.move_multistep(
                lift_steps,
                self.x_arr[-1],
                direction=2,  # tangential direction
                sign=-1,
            )

            self.transition_indices.append(len(self.x_arr) - 1)

            # Go to start point for next peeling cycle
            self.return_home_safe(distance_to_surface=self.retract_distance_to_surface)
            self.transition_indices.append(len(self.x_arr) - 1)
            print(f"Peeling period completed")

            # Move sideways for offseting the next peel
            self.move_multistep(
                self.num_slide_steps,
                self.x_arr[-1],
                direction=1,  # tangential direction
                sign=-1,
                project=True,
                distance_to_surface=self.retract_distance_to_surface,
            )
            self.transition_indices.append(len(self.x_arr) - 1)

            # Move back down to the surface
            self.move_multistep(
                lift_steps,
                self.x_arr[-1],
                direction=2,  # tangential direction
                sign=1,
            )
            self.transition_indices.append(len(self.x_arr) - 1)

            self.x_home = np.copy(self.x_arr[-1])

        self.trajectory = np.array(self.x_arr)
        self.trajectory_local_bases = np.array(self.trajectory_local_bases)

    def return_home_safe(self, distance_to_surface):
        local_basis_real = np.copy(self.pcloud.local_bases)

        u0 = np.zeros(len(self.pcloud.vertices))
        _, target_vertex = self.pcloud.get_closest_points(self.x_home)
        u0[self.source_vertices[0]] = 1
        geodesic_arr, geodesic_gradient_arr = (
            self.scalar_diffusion.precompute_geodesics_and_gradients(
                [self.source_vertices[0]]
            )
        )
        self.ut = self.scalar_diffusion.integrate_diffusion(u0)

        x_start = np.copy(self.x_arr[-1])
        for _ in range(500):
            x_next, local_basis = self.local_step(x_start, direction=0, sign=-1)

            x_next, _, projected_point = self.pcloud.correct_distance_smooth(
                x_next, distance_to_surface
            )
            geodesic_distance2home = geodesic_arr[0, projected_point]

            if geodesic_distance2home < 1e-2:
                break
            x_start = np.copy(x_next)
            # Append the next position and basis
            self.x_arr.append(x_next)

            self.pcloud.local_bases = local_basis_real
            # Get batch points for diffusion
            batch_points = self.wos.get_batch_from_point(x_next)
            # Compute local bases using diffusion
            local_basis, _, _ = self.wos.diffuse_rotations(batch_points)

            self.trajectory_local_bases.append(local_basis)

        self.pcloud.local_bases = local_basis_real
