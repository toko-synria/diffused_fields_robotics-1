"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import argparse

from diffused_fields.manifold import Pointcloud

from diffused_fields_robotics.local_action_primitives.action_primitives import Slicing


def main():
    parser = argparse.ArgumentParser(description="Slicing primitive on a point cloud.")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show interactive 3D animation in Polyscope (Play/Pause, speed slider).",
    )
    parser.add_argument(
        "--save-gif",
        metavar="PATH",
        default=None,
        help="Export trajectory animation to a GIF file (e.g. slicing.gif).",
    )
    parser.add_argument(
        "--mesh",
        default="banana_half.ply",
        help="Point cloud filename (default: banana_half.ply).",
    )
    args = parser.parse_args()

    pcloud = Pointcloud(filename=args.mesh)
    controller = Slicing(pcloud)
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
        controller.visualize_trajectory(show_tool=True, num_samples=5)


if __name__ == "__main__":
    main()
