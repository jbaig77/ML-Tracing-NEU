# GNN Adjacency Prediction from MATLAB Subgraphs

A GraphSAGE-based GNN that learns to predict edge connectivity (adjacency) in 3D neuron trace subgraphs. Takes `.mat` files containing node features and ground-truth adjacency matrices, builds a kNN proposal graph, and trains a model to predict which proposal edges are real.

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Data Format](#data-format)
4. [Environment Setup](#environment-setup)
5. [Running Locally](#running-locally)
6. [Running on the Cluster (SLURM)](#running-on-the-cluster-slurm)
7. [Parameter File Format](#parameter-file-format)
8. [All CLI Arguments](#all-cli-arguments)
9. [Model Architecture](#model-architecture)
10. [Outputs](#outputs)
11. [Resuming a Run](#resuming-a-run)
12. [Loop Penalty](#loop-penalty)
13. [Extending the Code](#extending-the-code)

---

## Overview

Each `.mat` file represents a small subgraph of a neuron trace. The model:

1. Reads node features `F` (xyz coords + optional patch/intensity features) and a ground-truth adjacency matrix `AM` from each `.mat`
2. Builds a kNN proposal graph on the 3D coordinates
3. Passes node features through a GraphSAGE GNN to get node embeddings
4. Predicts an edge probability for every proposal edge via an EdgeMLP
5. Trains with weighted binary cross-entropy (pos/neg class imbalance handled per-graph)
6. Optionally adds a triangle/loop penalty to discourage spurious cycles

---

## File Structure

```
.
├── train_gnn_adj_from_mat.py   # Main training script (model, data loading, eval, plotting)
├── run_gnn_adj_array.sh        # SLURM array job launcher
├── train_gnn_adj_params.txt    # Experiment grid (one row = one SLURM array task)
└── README.md
```

---

## Data Format

Each `.mat` file must contain exactly two variables:

| Variable | Shape     | Description                                                              |
|----------|-----------|--------------------------------------------------------------------------|
| `F`      | `(N, 3+P)` | Node features. First 3 columns are **xyz coordinates**. Remaining `P` columns are optional patch/intensity features. |
| `AM`     | `(N, N)`  | Ground-truth adjacency matrix. Values are binarized at 0.5 (i.e., any value > 0.5 is treated as a connected edge). Sparse scipy matrices are supported. |

**Directory layout expected under `DATA_ROOT`:**

```
DATA_ROOT/
├── train/
│   └── **/*.mat
└── validation/
    └── **/*.mat
```

Subdirectory names (`train`, `validation`) are configurable via `--train-subdir` and `--val-subdir`.

**Validation subset for metrics:** During training, only validation files whose filenames match `aug1` (but not `aug10`, `aug11`, etc.) are used for per-epoch metric logging and `.mat` export. This subset is referred to throughout as `val_aug1`.

---

## Environment Setup

The conda environment used is `traceGNN`. To recreate it manually:

```bash
conda create -n traceGNN python=3.10
conda activate traceGNN
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA version
pip install scipy matplotlib numpy
```

> No PyTorch Geometric or DGL required — message passing is implemented in pure PyTorch with `index_add_`.

---

## Running Locally

```bash
conda activate traceGNN

python train_gnn_adj_from_mat.py \
  --data-root /path/to/data \
  --epochs 100 \
  --lr 1e-3 \
  --nlayers 3 \
  --loop-penalty false \
  --loop-lambda 0.0 \
  --out-dir experiment_runs_gnn_adj \
  --run-name my_experiment \
  --seed 42 \
  --use-gpu true
```

---

## Running on the Cluster (SLURM)

The shell script `run_gnn_adj_array.sh` is a self-resubmitting SLURM array launcher. You always call it the same way regardless of how many experiments are in the param file:

```bash
sbatch run_gnn_adj_array.sh
```

It will:
1. Count the number of valid experiment lines in `train_gnn_adj_params.txt`
2. Resubmit itself as a proper SLURM array (`--array=0-N`) with the correct size
3. Each array task picks its corresponding line from the param file and launches one training run

**To use a different param file or output root, override via `--export`:**

```bash
sbatch --export=ALL,PARAM_FILE=my_other_params.txt,TOP_ROOT=my_output_dir run_gnn_adj_array.sh
```

**Default SLURM resource allocation** (edit the `#SBATCH` headers in the script to change):

| Resource         | Default     |
|------------------|-------------|
| CPUs per task    | 4           |
| Memory           | 100G        |
| Wall time        | 48 hours    |
| Partition        | short       |
| GPU              | None (CPU training by default in params; set `--use-gpu true` in params if GPU partition is used) |

**Logs** are written per-task to:
```
{TOP_ROOT}/{PARAM_FILE_BASENAME}/logs/output_{EXP_ID}_{JOB_ID}.out
{TOP_ROOT}/{PARAM_FILE_BASENAME}/logs/error_{EXP_ID}_{JOB_ID}.err
```

---

## Parameter File Format

`train_gnn_adj_params.txt` is a whitespace-delimited text file. Each non-comment, non-empty line defines one experiment. The columns must appear in this exact order:

```
# EXP_ID    DATA_ROOT    EPOCHS    LR    NLAYERS    LOOP_ON    LOOP_LAMBDA
```

| Column        | Type    | Description                                                     |
|---------------|---------|-----------------------------------------------------------------|
| `EXP_ID`      | string  | Unique experiment name. Used for output folder naming and logs. |
| `DATA_ROOT`   | path    | Absolute path to the dataset root (must contain `train/` and `validation/` subdirs). |
| `EPOCHS`      | int     | Number of training epochs.                                      |
| `LR`          | float   | Learning rate (e.g., `1e-3`).                                   |
| `NLAYERS`     | int     | Number of GraphSAGE layers.                                     |
| `LOOP_ON`     | 0 or 1  | Whether to enable the triangle loop penalty (`0` = off, `1` = on). |
| `LOOP_LAMBDA` | float   | Weight of the loop penalty loss term. Ignored if `LOOP_ON=0`.  |

Lines starting with `#` and blank lines are ignored.

**Example:**
```
L3_nopen    /data/traces    500    1e-3    3    0    0.0
L3_loop     /data/traces    500    1e-3    3    1    1e-2
```

The seed for each array task is automatically set as `4242 + SLURM_ARRAY_TASK_ID` so all runs are reproducible and distinct.

---

## All CLI Arguments

| Argument            | Type    | Default                  | Description                                                         |
|---------------------|---------|--------------------------|---------------------------------------------------------------------|
| `--data-root`       | str     | *(required)*             | Path to dataset root.                                               |
| `--train-subdir`    | str     | `train`                  | Subdirectory name for training `.mat` files.                        |
| `--val-subdir`      | str     | `validation`             | Subdirectory name for validation `.mat` files.                      |
| `--epochs`          | int     | `30`                     | Total training epochs.                                              |
| `--lr`              | float   | `1e-3`                   | Adam learning rate.                                                 |
| `--grad-clip`       | float   | `1.0`                    | Gradient norm clipping value.                                       |
| `--thresh`          | float   | `0.5`                    | Sigmoid threshold for binary edge prediction.                       |
| `--hidden-dim`      | int     | `128`                    | Hidden dimension for GNN layers and EdgeMLP.                        |
| `--nlayers`         | int     | `2`                      | Number of GraphSAGE message-passing layers.                         |
| `--dropout`         | float   | `0.1`                    | Dropout rate throughout the model.                                  |
| `--knn-k`           | int     | `16`                     | Number of nearest neighbors for proposal graph construction.        |
| `--loop-penalty`    | bool    | `false`                  | Enable triangle loop penalty.                                       |
| `--loop-lambda`     | float   | `0.0`                    | Weight of loop penalty term.                                        |
| `--loop-hutch-R`    | int     | `8`                      | Number of Hutchinson samples for trace estimation in loop penalty.  |
| `--out-dir`         | str     | `outputs_gnn_adj`        | Base output directory.                                              |
| `--run-name`        | str     | `None`                   | Run name prefix. Output folder becomes `{out-dir}/{run-name}_{timestamp}`. |
| `--resume-if-found` | bool    | `true`                   | Resume from latest matching checkpoint if one exists.               |
| `--seed`            | int     | `42`                     | Random seed.                                                        |
| `--use-gpu`         | bool    | `true`                   | Use CUDA if available.                                              |

---

## Model Architecture

```
Input: F (N, 3+P)
  │
  ├─ coords (N, 3) ──────────────────────────────────────────────┐
  │                                                               │
  └─ patch features (N, P) → PatchMLP(P → 64) ──────────────────┤
                                                                  ↓
                                              InProj: Linear(3+64, hidden_dim) → ReLU → Dropout
                                                                  │
                                              [GraphSAGE Layer] × nlayers
                                               (mean aggregation, pure PyTorch)
                                                                  │
                                              Node embeddings H (N, hidden_dim)
                                                                  │
                                  For each proposal edge (i, j):
                                  EdgeMLP([h_i, h_j, |h_i - h_j|, Δxyz]) → scalar logit
                                                                  │
                                              Sigmoid → edge probability p_ij
```

**Loss:** Weighted BCE per graph (`pos_weight = neg_count / pos_count`) + optional triangle penalty.

**Optimizer:** Adam with `weight_decay=1e-4`.

---

## Outputs

All outputs for a run land in `{out_dir}/{run_name}_{timestamp}/`:

```
{run_dir}/
├── config_resolved.json                    # Full config snapshot for the run
├── checkpoint_latest.pt                    # Latest model + optimizer + history checkpoint
├── init_model_state.pt                     # Saved initial weights (for epoch-0 baseline; created once)
├── baseline_epoch0_done.txt                # Flag file: prevents duplicate epoch-0 baseline computation
│
├── figs/
│   ├── dashboard_epoch_{NNN}.png                      # 6-panel (train+val, loss/acc/ba)
│   ├── dashboard_combined_epoch_{NNN}.png             # 3-panel combined (train vs val)
│   ├── dashboard_recent75_epoch_{NNN}.png             # Last 75 epochs, no smoothing
│   ├── dashboard_recent75_smoothed_epoch_{NNN}.png    # Last 75 epochs, EMA smoothed
│   ├── dashboard_first80_loss_ba_epoch_{NNN}.png      # First 80 epochs, loss + BA only
│   └── val_trace_epoch_{NNN}.png                      # Side-by-side GT vs predicted trace (top-down XY)
│
└── validation_aug1_epoch{NNN}/
    └── {original_filename}.mat             # Per-file predictions (A_prob, AM_gt, r, p_edge, edge_index)
```

**Saved `.mat` fields per validation file:**

| Field         | Description                                              |
|---------------|----------------------------------------------------------|
| `A_prob`      | Dense `(N, N)` soft adjacency (symmetric, edge probabilities) |
| `AM_gt`       | Ground-truth adjacency matrix                           |
| `r`           | Node coordinates `(N, 3)`                               |
| `p_edge`      | Per-edge probabilities `(E,)` on the proposal edges     |
| `edge_index`  | Proposal edge index `(2, E)` (undirected, i < j)        |
| `source_path` | Path to the original input `.mat` file                  |

---

## Resuming a Run

Resumption is automatic. If `--resume-if-found true` (the default) and a `checkpoint_latest.pt` exists in the run directory, training picks up from the last completed epoch. The run directory is matched by `{out_dir}/{run_name}_*` — the latest timestamped folder is used.

To start fresh even if a checkpoint exists, either use a new `--run-name` or pass `--resume-if-found false`.

The epoch-0 baseline (initial model metrics prepended to history) is also protected by a flag file and will not be recomputed on resume.

---

## Loop Penalty

When `--loop-penalty true` and `--loop-lambda > 0`, a differentiable triangle-count penalty is added to the loss:

```
loss = BCE + loop_lambda * triangle_penalty
```

The triangle count is estimated via **Hutchinson trace estimation** of `trace(A³)/6` using `--loop-hutch-R` random ±1 vectors. This avoids materializing the full `N×N` adjacency during training and scales to large graphs. The penalty is normalized by `(N + edge_mass)` to be roughly scale-invariant.

The current experiment grid in `train_gnn_adj_params.txt` sweeps:
- `loop_lambda` in `{0, 1e-2, 1e-1, 1e0, 1e1}`
- `nlayers` in `{1, 2, 3, 5, 10}`

---

## Extending the Code

**Adding a new loss term:** Add it in the training loop and `eval_paths` function after the BCE computation. Both must stay in sync.

**Changing the proposal graph:** Replace `build_knn_edges` with any function returning a `(2, E)` undirected edge index with `i < j`.

**Adding node features:** Extend the `F` matrix in your `.mat` files beyond column 3. The `patch_dim` is inferred automatically from the first valid training file.

**Adding a learning rate scheduler:** The optimizer is created in `train()`. Add a scheduler after the `opt = torch.optim.Adam(...)` line and step it per epoch.

**Changing the validation subset logic:** The `is_aug1_file()` function controls which validation files are used for per-epoch metric logging and `.mat` export. Modify the regex there to match a different naming convention.