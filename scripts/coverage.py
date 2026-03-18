"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import argparse

from diffused_fields.manifold import Pointcloud

from diffused_fields_robotics.local_action_primitives.action_primitives import Coverage


def main():
    parser = argparse.ArgumentParser(description="Coverage primitive on a point cloud.")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show interactive 3D animation in Polyscope (Play/Pause, speed slider).",
    )
    parser.add_argument(
        "--save-gif",
        metavar="PATH",
        default=None,
        help="Export trajectory animation to a GIF file (e.g. coverage.gif).",
    )
    parser.add_argument(
        "--mesh",
        default="stiffness_sample.ply",
        help="Point cloud filename (default: stiffness_sample.ply).",
    )
    parser.add_argument(
        "--diffusion-scalar",
        type=float,
        default=10,
        help="Diffusion scalar (default: 10).",
    )
    args = parser.parse_args()

    pcloud = Pointcloud(filename=args.mesh)
    controller = Coverage(pcloud, diffusion_scalar=args.diffusion_scalar)
    controller.run()

    if args.save_gif:
        controller.visualize_trajectory(
            show_tool=True,
            num_samples=None,
            save_animation=args.save_gif,
        )
    elif args.animate:
        controller.visualize_trajectory(show_tool=True, num_samples=None)
    else:
        controller.visualize_trajectory(show_tool=False)


if __name__ == "__main__":
    main()
