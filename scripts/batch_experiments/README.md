# Batch Experiments

This directory contains batch experiment scripts for evaluating action primitives under various noise conditions and deformations.

## Available Experiments

### 1. Transfer across Objects Experiments
**File:** `batch_peeling.py`

Runs peeling experiments with random point cloud deformations (scaling, bending, and twisting). We use 10 experiments as default for small file size and fast execution. For the paper we used 50 experiments.

**Features:**
- Random scaling per axis (0.5-1.5x)
- Random bending (curvature range: -3.0 to 3.0)
- Random twisting (strength range: -3.0 to 3.0)
- Body-fixed coordinate system computation
- Ground truth transition tracking

**Usage:**
```bash
# Run all experiments and save results
python scripts/batch_experiments/batch_peeling.py

# Run and visualize one specific experiment (e.g. exp_idx=0, sample_idx=0)
python scripts/batch_experiments/batch_peeling.py --animate-exp 0 --animate-sample 0

# Run and export a GIF for a specific experiment
python scripts/batch_experiments/batch_peeling.py --animate-exp 0 --save-gif peeling_batch_exp0.gif
```
Outputs a pickled experiment results file and, if requested, an animation or GIF for the selected experiment.

**Analysis:**
Compare the results in terms of trajectory statistics:
1. Compare trajectories expressed in Cartesian, cylindirical, spherical and local (ours) reference frames
```bash
python scripts/analysis/batch_peeling_stats_primitives.py
```
2. Compare trajectories expressed in sampled frames across the point cloud. Discrete approximation of our method.
```bash
python scripts/analysis/batch_peeling_stats_sampled_frames.py
```

Output plots are stored in `/results/batch_experiments/`


### 2. Robustness to Input Noise Experiments

Evaluates slicing robustness under different types of input noise:
- **`batch_slicing_geometric_noise.py`**: Gaussian noise on point cloud vertices
- **`batch_slicing_topological_noise.py`**: Missing data (holes in point cloud)
- **`batch_slicing_keypoint_noise.py`**: Noise in source vertex positions

All experiments test across multiple diffusion scalars (0.1 to 10000) with 50 noise realizations each.

**Usage:**
```bash
python scripts/batch_experiments/batch_slicing_geometric_noise.py
python scripts/batch_experiments/batch_slicing_topological_noise.py
python scripts/batch_experiments/batch_slicing_keypoint_noise.py
```

**Analysis:**
```bash
python scripts/analysis/robustness.py
```

Output plots are stored in `/results/batch_experiments/`

## Notes

- All experiments save results as pickle files in the project results directory
- Peeling experiments use `pear.ply` object
- Slicing experiments use `banana_half.ply` object
- Diffusion scalars sweep: `np.logspace(np.log10(0.1), np.log10(10000), 10)`
