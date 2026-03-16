# Diffused Fields Robotics

This package is supplementary material for the paper **"Object-centric Task Representation and Transfer using Diffused Orientation Fields"**.


This is the **robotics package** for object-centric robot manipulation applications: local action primitives (peeling, slicing and tactile coverage), trajectory optimization, and reinforcement learning using **Diffused Orienation Fields (DOF)**. 

This package depends on the **[diffused_fields](https://github.com/idiap/diffused_fields)** for computing **DOF** on point clouds.

## Installation

## Large files (Git LFS required)

This repository uses [Git LFS](https://git-lfs.github.com/) to store large files
(e.g. data, models, point clouds).

### 1. Before cloning

Make sure Git LFS is installed **before** you clone. You can skip this step if you already have it installed and activated in your machine.

Install Git LFS (Ubuntu)
```bash
sudo apt install git-lfs
```
Install Git LFS (macOS) using homebrew
```bash
brew install git-lfs
```
run once to enable LFS
```bash
git lfs install 
```
This package depends on the `diffused_fields` library. First clone both repositories:

```bash
git clone https://github.com/idiap/diffused_fields.git
git clone https://github.com/idiap/diffused_fields_robotics.git
```

Create a virtual environment and install both packages in editable mode:
```bash
cd diffused_fields_robotics
# Create a virtual environment using Python 3.12
python3.12 -m venv df
# Activate the virtual environment
source df/bin/activate
# Install diffused_fields in editable mode using path to its root directory
pip install -e ../diffused_fields
# Install diffused_fields_robotics in editable mode
pip install -e .
```


## Paper and Citation

If you use this package in your research, please cite: (Coming soon)


## Quick Start

### Running Action Primitives

```bash
# Run slicing on a banana
python scripts/slicing.py
# With interactive animation:  python scripts/slicing.py --animate
# Export to GIF:               python scripts/slicing.py --save-gif slicing.gif

# Run peeling on a pear
python scripts/peeling.py
# With interactive animation:  python scripts/peeling.py --animate
# Export to GIF:               python scripts/peeling.py --save-gif peeling.gif

# Run coverage on a surface
python scripts/coverage.py
```

### Running Batch Experiments

See [scripts/batch_experiments/README.md](scripts/batch_experiments/README.md) for details.

**Transfer across objects:**
```bash
python scripts/batch_experiments/batch_peeling.py
python scripts/analysis/batch_peeling_stats_primitives.py
```

**Robustness to noise:**
```bash
python scripts/batch_experiments/batch_slicing_geometric_noise.py
python scripts/analysis/robustness.py
```

## Reproducing Paper Results

All simulation data and plots from the paper can be generated using the scripts in `scripts/batch_experiments/` and `scripts/analysis/`.

---

This code is maintained by Cem Bilaloglu and licensed under the MIT License.

Copyright (c) 2025 Idiap Research Institute - contact@idiap.ch