# ML-Tracing

# Automated Neurite Tracing with Graph Neural Networks

> **Status:** Work in progress — unpublished research project.

This project uses graph neural networks to automate the tracing of neurites (axons and dendrites) from 3D microscopy image data. Given a set of candidate nodes (seeds) extracted from image patches, the goal is to predict which pairs of nodes should be connected — effectively reconstructing the underlying neurite topology as a graph.

Two approaches are implemented, each in its own folder:

| Folder | Approach | Entry point |
|--------|----------|-------------|
| `GNN/` | Simple single-pass GNN (GraphSAGE + EdgeMLP) | `train_gnn_adj_from_mat.py` |
| `DGNN/` | Diffusion-based iterative refinement (Transformer + reverse diffusion) | `train_gnn4.py` |

Both approaches share the same core idea: nodes carry 3D coordinates and optional patch/intensity features extracted from the image, edges are proposed by kNN on coordinates, and the model predicts edge probabilities to reconstruct the true adjacency.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Environment Setup](#environment-setup)
3. [Data Format](#data-format)
4. [GNN — Single-Pass Approach](#gnn--single-pass-approach)
   - [How It Works](#how-it-works)
   - [Running Locally](#running-locally-gnn)
   - [Running on the Cluster (SLURM)](#running-on-the-cluster-slurm-gnn)
   - [Parameter File Format](#parameter-file-format)
   - [CLI Arguments](#cli-arguments-gnn)
   - [Outputs](#outputs-gnn)
5. [DGNN — Diffusion-Based Approach](#dgnn--diffusion-based-approach)
   - [How It Works](#how-it-works-dgnn)
   - [Running Locally](#running-locally-dgnn)
   - [CLI Arguments](#cli-arguments-dgnn)
   - [Outputs](#outputs-dgnn)
6. [Shared Concepts](#shared-concepts)
   - [Loop / Cycle Penalties](#loop--cycle-penalties)
   - [Resuming Runs](#resuming-runs)
   - [Epoch-0 Baseline](#epoch-0-baseline)
7. [Extending the Code](#extending-the-code)

---

## Repository Layout

```
.
├── MATLAB/                             # Preprocessing pipeline (run first)
│
├── GNN/
│
└── DGNN/
```

---

## MATLAB Preprocessing

> **Run this before any Python training.** The MATLAB pipeline converts raw 3D microscopy image volumes into the graph-structured inputs consumed by the GNN and DGNN training scripts.

The `MATLAB/` folder contains the preprocessing pipeline. It performs three things in sequence:

**1. Seed node extraction and patch extraction**
Candidate neurite locations (seed nodes) are identified in the image volume. For each seed, a local 3D image patch is extracted and summarized into a feature vector. These features form the columns of `F` beyond the first three (xyz) coordinates.

**2. Subgraph construction and `.mat` export**
For each seed node neighborhood, a subgraph is assembled: the feature matrix `F` (xyz + patch features) and the ground-truth adjacency matrix `AM` are saved together as a `.mat` file. These are the files consumed by the `GNN/` training script.

**3. Augmentation**
Each subgraph is saved in multiple augmented variants. The augmentation index is encoded in the filename — `aug1`, `aug2`, etc. The suffix `aug1` specifically identifies the unaugmented (or minimally augmented) reference copy, which is why both training scripts use `aug1` files for per-epoch validation export and qualitative inspection. Files named `aug10`, `aug11`, etc. are treated as separate augmented variants and are not confused with `aug1` due to the regex match `aug1(?!\d)`.

**Outputs of the MATLAB pipeline:**

```
DATA_ROOT/
├── train/
│   └── **/*_aug1.mat, *_aug2.mat, ...
└── validation/
    └── **/*_aug1.mat, *_aug2.mat, ...
```

The `.mat` files output here are passed directly as `DATA_ROOT` to `train_gnn_adj_from_mat.py`. For the DGNN pipeline, an additional conversion step (not yet documented) is needed to produce `.pt` files from these `.mat` subgraphs.

> **Note:** The MATLAB preprocessing scripts are not yet fully documented in this README. If you are picking up this project, check the `MATLAB/` folder for script names and any inline comments describing parameters such as patch size, seed detection thresholds, and augmentation types.

---

## Environment Setup

Both scripts share the same dependencies. The conda environment used for this project is `traceGNN`.

> **Note on Python version:** The exact Python version used during initial development is **Python 3.7**.

### Option 1 — Recreate from scratch

```bash
conda create -n traceGNN python=3.8
conda activate traceGNN

# PyTorch — adjust the CUDA version to match your cluster/machine
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Core dependencies
pip install scipy matplotlib numpy

# DGNN only — needed for .pt file loading and DataLoader
# (torch is already installed above; no PyG or DGL required)
```

> **No PyTorch Geometric or DGL required.** All message passing in both scripts is implemented in pure PyTorch using `index_add_`.

### Option 2 — Quick pip install list

```
torch>=1.12
torchvision
scipy
matplotlib
numpy
```

### Verify your install

```bash
conda activate traceGNN
python -c "import torch, scipy, matplotlib, numpy; print('torch', torch.__version__)"
```

---

## Data Format

### GNN (`.mat` files)

Each `.mat` file represents one subgraph. It must contain:

| Variable | Shape     | Description |
|----------|-----------|-------------|
| `F`      | `(N, 3+P)` | Node features. First 3 columns are **xyz coordinates**. Remaining `P` columns are patch/intensity features (can be 0). |
| `AM`     | `(N, N)`  | Ground-truth adjacency matrix. Binarized at 0.5. Scipy sparse matrices are supported. |

**Directory layout under `DATA_ROOT`:**
```
DATA_ROOT/
├── train/
│   └── **/*.mat
└── validation/
    └── **/*.mat
```

### DGNN (`.pt` files)

Each `.pt` file is a PyTorch graph object with attributes:

| Attribute | Description |
|-----------|-------------|
| `x` | `(N, F)` node features — first 3 cols are xyz coordinates |
| `edge_index` | `(2, E)` proposed edges |
| `edge_label` | `(E,)` binary ground-truth edge labels |
| `max_noise_edge_index` | *(optional)* `(2, Emax)` edges surviving at max diffusion noise — used to initialize inference |
| `R` | *(optional)* `(E,)` survival scores for edges — fallback init if `max_noise_edge_index` absent |
| `true_positions` | *(optional)* `(N, 3)` original voxel coordinates for visualization |

**Directory layout under `DATA_ROOT`:**
```
DATA_ROOT/
├── run1_full_pt/               # (run_prefix)
│   ├── train/
│   │   └── **/*.pt
│   └── validation/
│       └── **/*.pt
```

The `run_prefix` (default: `run1_full_pt`) is configurable and used to identify which run subdirectories to scan.

### Validation subset for per-epoch logging

Both approaches filter validation files whose filenames match `aug1` (but **not** `aug10`, `aug11`, etc.) for per-epoch `.mat` export and metric logging. This naming convention reflects the augmentation scheme used to generate the data. Files named `aug1_*.mat` or `*_aug1.pt` will be picked up; `*aug10*` and `*aug11*` will not.

---

## GNN — Single-Pass Approach

### How It Works

The GNN takes a single forward pass per training step. For each `.mat` subgraph:

1. Build a kNN proposal graph on xyz coordinates
2. Encode node features through a `PatchMLP` (if patch features exist) and a linear projection
3. Run `N` layers of **GraphSAGE** (mean aggregation, pure PyTorch)
4. For each proposed edge `(i, j)`, concatenate `[h_i, h_j, |h_i - h_j|, Δxyz]` and score through an **EdgeMLP**
5. Train with **class-weighted BCE** (pos/neg imbalance handled per-graph)
6. Optionally add a **triangle loop penalty** to discourage spurious cycles

**Architecture summary:**
```
F (N, 3+P)
  ├── coords (N,3) → z-score
  └── patch features (N,P) → PatchMLP(P→64)
           ↓
     InProj: Linear(3+64, hidden_dim) → ReLU → Dropout
           ↓
     GraphSAGELayer × nlayers
           ↓
     Node embeddings H (N, hidden_dim)
           ↓
     EdgeMLP([h_i, h_j, |h_i-h_j|, Δxyz]) → logit per edge
           ↓
     Sigmoid → edge probability
```

### Running Locally (GNN)

```bash
conda activate traceGNN

python GNN/train_gnn_adj_from_mat.py \
  --data-root /path/to/data \
  --epochs 500 \
  --lr 1e-3 \
  --nlayers 3 \
  --loop-penalty false \
  --loop-lambda 0.0 \
  --out-dir experiment_runs_gnn_adj \
  --run-name my_experiment \
  --seed 42 \
  --use-gpu false
```

### Running on the Cluster (SLURM, GNN)

The shell script `run_gnn_adj_array.sh` is **self-resubmitting** — it reads the param file, counts how many experiments there are, and re-launches itself as a properly sized SLURM array. You always invoke it the same way:

```bash
cd GNN/
sbatch run_gnn_adj_array.sh
```

To use a different param file or output root:
```bash
sbatch --export=ALL,PARAM_FILE=my_other_params.txt,TOP_ROOT=my_output_dir run_gnn_adj_array.sh
```

**Default resource allocation** (edit `#SBATCH` headers in the script to change):

| Resource | Default |
|----------|---------|
| CPUs per task | 4 |
| Memory | 100G |
| Wall time | 48 hours |
| Partition | `short` |
| GPU | None (CPU training by default) |

Logs are written to:
```
{TOP_ROOT}/{PARAM_FILE_BASENAME}/logs/output_{EXP_ID}_{JOB_ID}.out
{TOP_ROOT}/{PARAM_FILE_BASENAME}/logs/error_{EXP_ID}_{JOB_ID}.err
```

### Parameter File Format

`train_gnn_adj_params.txt` is a whitespace-delimited file. Each non-comment, non-blank line defines one experiment (one SLURM array task). Columns must appear in this order:

```
# EXP_ID    DATA_ROOT    EPOCHS    LR    NLAYERS    LOOP_ON    LOOP_LAMBDA
```

| Column | Type | Description |
|--------|------|-------------|
| `EXP_ID` | string | Unique experiment name used for output folder naming and logs |
| `DATA_ROOT` | path | Absolute path to the dataset root |
| `EPOCHS` | int | Number of training epochs |
| `LR` | float | Learning rate (e.g. `1e-3`) |
| `NLAYERS` | int | Number of GraphSAGE layers |
| `LOOP_ON` | 0 or 1 | Enable triangle loop penalty |
| `LOOP_LAMBDA` | float | Weight of loop penalty. Ignored if `LOOP_ON=0` |

Lines beginning with `#` and blank lines are ignored. Seeds are set automatically as `4242 + SLURM_ARRAY_TASK_ID`.

**Current experiment grid** sweeps over:
- `nlayers` ∈ {1, 2, 3, 5, 10}
- `loop_lambda` ∈ {0, 1e-2, 1e-1, 1e0, 1e1}

### CLI Arguments (GNN)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-root` | str | *(required)* | Path to dataset root |
| `--train-subdir` | str | `train` | Subdir name for training `.mat` files |
| `--val-subdir` | str | `validation` | Subdir name for validation `.mat` files |
| `--epochs` | int | `30` | Training epochs |
| `--lr` | float | `1e-3` | Adam learning rate |
| `--grad-clip` | float | `1.0` | Gradient norm clipping |
| `--thresh` | float | `0.5` | Sigmoid threshold for binary prediction |
| `--hidden-dim` | int | `128` | Hidden dimension for GNN and EdgeMLP |
| `--nlayers` | int | `2` | Number of GraphSAGE layers |
| `--dropout` | float | `0.1` | Dropout rate |
| `--knn-k` | int | `16` | kNN neighbors for proposal graph |
| `--loop-penalty` | bool | `false` | Enable triangle loop penalty |
| `--loop-lambda` | float | `0.0` | Loop penalty weight |
| `--loop-hutch-R` | int | `8` | Hutchinson samples for loop penalty estimation |
| `--out-dir` | str | `outputs_gnn_adj` | Base output directory |
| `--run-name` | str | `None` | Run name; output folder becomes `{out-dir}/{run-name}_{timestamp}` |
| `--resume-if-found` | bool | `true` | Resume from latest checkpoint if found |
| `--seed` | int | `42` | Random seed |
| `--use-gpu` | bool | `true` | Use CUDA if available |

### Outputs (GNN)

```
{out_dir}/{run_name}_{timestamp}/
├── config_resolved.json
├── checkpoint_latest.pt
├── init_model_state.pt                         # Saved initial weights for epoch-0 baseline
├── baseline_epoch0_done.txt                    # Flag: prevents duplicate epoch-0 computation
│
├── figs/
│   ├── dashboard_epoch_{NNN}.png               # 6-panel (train + val: loss, acc, BA)
│   ├── dashboard_combined_epoch_{NNN}.png      # 3-panel combined (train vs val)
│   ├── dashboard_recent75_epoch_{NNN}.png      # Last 75 epochs
│   ├── dashboard_recent75_smoothed_epoch_{NNN}.png   # Last 75 epochs, EMA smoothed
│   ├── dashboard_first80_loss_ba_epoch_{NNN}.png     # First 80 epochs, loss + BA only
│   └── val_trace_epoch_{NNN}.png              # GT vs predicted trace, XY top-down view
│
└── validation_aug1_epoch{NNN}/
    └── {filename}.mat                          # Per-file prediction outputs
```

**Saved `.mat` fields:**

| Field | Description |
|-------|-------------|
| `A_prob` | `(N,N)` soft adjacency (edge probabilities, symmetric) |
| `AM_gt` | Ground-truth adjacency matrix |
| `r` | Node coordinates `(N,3)` |
| `p_edge` | Per-edge probabilities `(E,)` |
| `edge_index` | Proposal edges `(2,E)`, undirected with i < j |
| `source_path` | Path to the original `.mat` input file |

---

## DGNN — Diffusion-Based Approach

### How It Works

Rather than making a single prediction, DGNN iteratively refines edge probabilities through a **reverse diffusion process** inspired by score-matching / denoising diffusion. The key idea is that a noisy version of the ground-truth edge labels is progressively corrupted, and the model learns to denoise it.

**Training (teacher-forced):**
1. Start from ground-truth edge labels `p0`
2. Apply a cosine noise schedule for `t` steps, blending `p0` toward a prior `w` (based on geometry and image appearance similarity)
3. Feed the noisy `p_t` plus node features into the model at timestep `t`
4. Predict the denoised edge probabilities and supervise with BCE/focal loss

**Inference (reverse diffusion):**
1. Initialize from the `max_noise_edge_index` if available (edges that survived maximum corruption), or fall back to the geometric/appearance prior `w`
2. Iteratively apply the trained model along a reverse schedule, damping the update at each step: `p_next = (1 - damping) * p_in + damping * p_hat`

**Optional free-run training:** With probability `free_run_prob`, instead of teacher forcing, the model unrolls `free_run_steps` steps from the inference-style initialization to reduce the train/inference distribution gap.

**Architecture (`baseline` arch):**
```
x_all (N, 3+P)
  ├── coords → z-score
  ├── patch features → NodePatchEncoder(P → 64)
  └── sinusoidal time embedding (t → 32-dim)
           ↓
     Linear projection → d_model
           ↓
     TransformerEncoder × nlayers   (node-level self-attention)
           ↓
     Node embeddings H (N, d_model)
           ↓
     Per-edge: [p_in, dist_z, sim_01, Δxyz, h_i, h_j] → EdgeMLP → sigmoid
```

**Additional architectures (selectable via `--arch`):**

| `--arch` | Description |
|----------|-------------|
| `baseline` | Node Transformer + EdgeMLP (default) |
| `graphaware` | Same as baseline but node self-attention is masked to kNN neighbors |
| `edgecomp` | Adds edge-level self-attention between edges sharing a node (line-graph style) |
| `temporal` | Adds cross-attention from edge tokens to a memory of {w, p_in} |
| `combo` | edgecomp + temporal stacked |
| `edge_mlp` | Minimal MLP over handcrafted edge features only (no node encoder) |
| `edge_mlp_attn` | Tiny edge-token self-attention without a full node encoder |

### Running Locally (DGNN)

```bash
conda activate traceGNN

python DGNN/train_gnn4.py \
  --data-root /path/to/data \
  --run-prefix run1_full_pt \
  --epochs 100 \
  --lr 1e-3 \
  --arch baseline \
  --nlayers 2 \
  --T 10 \
  --loss-type focal \
  --loop-penalty none \
  --out-dir experiment_runs_dgnn \
  --run-name my_diffusion_experiment \
  --seed 42
```

### CLI Arguments (DGNN)

**Data:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | *(required)* | Path to dataset root |
| `--train-subdir` | `train` | Subdir name for training `.pt` files |
| `--val-subdir` | `validation` | Subdir name for validation `.pt` files |
| `--run-prefix` | `run1_full_pt` | Prefix of run subdirectories to scan for `.pt` files |

**Diffusion:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--T` | `10` | Number of diffusion timesteps |
| `--s-min` | `0.05` | Minimum noise level (cosine schedule) |
| `--s-max` | `0.7` | Maximum noise level |
| `--jitter-std` | `0.05` | Gaussian jitter added during corruption |
| `--inference-mode` | `max_noise` | Initialization for inference: `max_noise` or `prior_only` |
| `--val-infer-steps` | `None` (uses T) | Number of reverse steps at evaluation |
| `--val-damping` | `0.6` | Damping factor for reverse step: `p = (1-d)*p_in + d*p_hat` |

**Training:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `30` | Training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch-graphs` | `1` | Graphs per batch |
| `--grad-clip` | `1.0` | Gradient clipping |
| `--loss-type` | `focal` | Loss function: `bce` or `focal` |
| `--focal-gamma` | `2.0` | Focal loss gamma |
| `--use-curriculum` | `true` | Curriculum: anneal minimum t from high to low |
| `--start-min-t` | `8` | Starting minimum t (start of training) |
| `--end-min-t` | `1` | Ending minimum t (end of training) |
| `--p-in-dropout` | `0.5` | Probability of replacing p_in entries with prior w (augmentation) |
| `--p-in-noise-std` | `0.20` | Gaussian noise added to p_in (augmentation) |
| `--free-run-prob` | `0.3` | Probability of using free-run (inference-style) training instead of teacher forcing |
| `--free-run-steps` | `2` | Number of unrolled steps during free-run training |
| `--free-run-damping` | `0.6` | Damping for free-run update steps |

**Architecture:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `baseline` | Model architecture (see table above) |
| `--d-model` | `128` | Transformer hidden dimension |
| `--nhead` | `8` | Number of attention heads |
| `--nlayers` | `2` | Transformer encoder layers |
| `--time-dim` | `32` | Sinusoidal time embedding dimension |
| `--edge-head-hidden` | `128` | EdgeMLP hidden size |
| `--graph-k` | `16` | kNN neighbors for graph-aware attention masking |

**Loop / cycle penalties:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--loop-penalty` | `none` | Penalty type: `none`, `simple`, `complex`, or `nbt` |
| `--loop-lambda` | `0.0` | Loop penalty weight |
| `--loop-K` | `6` | Extra powers in truncated series (for `complex` and `nbt`) |
| `--loop-tau` | `1.1` | Resolvent parameter τ for `complex` penalty |
| `--loop-complex-estimator` | `auto` | Estimator: `auto`, `exact`, or `hutch` |
| `--loop-hutch-R` | `8` | Hutchinson probe count |
| `--loop-nbt-weighting` | `sqrt` | Edge weighting for `nbt`: `sqrt` or `child` |

**Endpoint penalties:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--endpoint-penalty` | `none` | Penalty: `none`, `bump`, `deg2`, or `quad` |
| `--endpoint-lambda` | `0.0` | Endpoint penalty weight |
| `--endpoint-sigma` | `0.5` | Width of degree≈1 Gaussian bump (for `bump`) |
| `--endpoint-mask-gt` | `true` | Only apply endpoint penalty on GT-touched nodes |

**IO:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--out-dir` | `outputs_graph_diffusion_october` | Base output directory |
| `--run-name` | `None` | Run name prefix; timestamp appended |
| `--resume-if-found` | `true` | Resume from latest checkpoint |
| `--seed` | `None` | Random seed |
| `--thresh` | `0.5` | Sigmoid threshold for metrics |

### Outputs (DGNN)

```
{out_dir}/{run_name}_{timestamp}/
├── config_resolved.json
├── checkpoint_latest.pt
├── best_model.pt                                  # Best checkpoint by val BA
├── init_model_state.pt
├── baseline_epoch0_done.txt
│
├── figs/
│   ├── dashboard_epoch_{NNN}.png                 # 9-panel (train, train-from-max, val)
│   ├── dashboard_combined_epoch_{NNN}.png        # 3-panel: loss, acc, BA (train vs val)
│   ├── dashboard_last10_epoch_{NNN}.png          # Last 10 epochs, auto y-lim
│   ├── qual_epoch_{NNN}.png                      # MIP-style edge overlay (pred/GT/overlay)
│   └── qual_epoch_trace_{NNN}.png               # 3D trace: GT vs predicted, XY top view
│
└── validation_on_full_image_epoch{NNN}/
    └── {filename}.mat
```

**Saved `.mat` fields:**

| Field | Description |
|-------|-------------|
| `A_prob` | `(N,N)` soft adjacency |
| `r` | Node coordinates `(N,3)` |
| `p_edge` | Per-edge probabilities `(E,)` |
| `edge_index` | `(2,E)` proposal edges |
| `pt_path` | Path to source `.pt` file |

---

## Shared Concepts

### Loop / Cycle Penalties

Both scripts support optional penalties to suppress spurious cycles in the predicted graph. All penalties operate on **soft (unthresholded) edge probabilities** and are differentiable.

| Penalty | Description | Available in |
|---------|-------------|--------------|
| `simple` (GNN) / `simple` (DGNN) | Estimates `trace(A³)/6` via Hutchinson — penalizes triangle mass | Both |
| `complex` | Resolvent-based: `trace(A³ (τI - A)⁻¹)` — penalizes all closed walks with geometric discount | DGNN only |
| `nbt` | Non-backtracking (Hashimoto) operator — penalizes closed walks while ignoring immediate edge reversals | DGNN only |

All penalties are normalized by `(N + edge_mass)` to remain approximately scale-invariant across graph sizes.

### Resuming Runs

Both scripts support automatic resumption. If a `checkpoint_latest.pt` exists in the run directory and `--resume-if-found true` (the default), training continues from the last completed epoch. The run directory is matched by `{out_dir}/{run_name}_*` — the latest timestamped folder wins.

To start fresh: use a new `--run-name`, or pass `--resume-if-found false`.

### Epoch-0 Baseline

Both scripts compute a **pre-training baseline** (epoch 0) the first time a run folder is created, and prepend these metrics to the history so all training curves start from a principled zero-shot reference point. This is gated by a `baseline_epoch0_done.txt` flag file and will not rerun on resume.

---

## Extending the Code

**Adding a new data source (GNN):** The data loader expects `F` and `AM` keys in each `.mat`. Any preprocessing that produces those two arrays and saves to `.mat` is compatible.

**Adding a new architecture (DGNN):** Add a new branch in `build_model()` and `model_factory()` in `train_gnn4.py`, and register the arch name in the `--arch` argparse choices.

**Adding a new loop penalty:** In both scripts, add the penalty computation in the training loop (after the base loss) and in `eval_paths` / `eval_max_rollout_paths` to keep train and eval metrics consistent.

**Adding a learning rate scheduler (GNN):** The optimizer is created in `train()`. Add a `ReduceLROnPlateau` or cosine annealer immediately after `opt = torch.optim.Adam(...)` and step it at the end of each epoch. The DGNN script already does this (tracking `val_loss`).

**Changing the proposal graph:** In the GNN, replace `build_knn_edges` with any function returning `(2, E)` undirected edges with `i < j`. In the DGNN, the proposal graph comes pre-built in the `.pt` files via `edge_index`.
