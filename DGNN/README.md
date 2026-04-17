# TracingGNN — Graph Diffusion Training Pipeline

A graph neural network training pipeline for neuron/vascular tracing from 3D fluorescence/multiphoton microscopy. The model learns to predict edge probabilities in a candidate graph built over detected seed points, using a diffusion-based corruption/denoising framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Files](#repository-files)
3. [Environment Setup](#environment-setup)
4. [Data Format](#data-format)
5. [Model Architectures](#model-architectures)
6. [Training Concepts](#training-concepts)
7. [Running Experiments](#running-experiments)
8. [Parameter File Reference](#parameter-file-reference)
9. [Runner Script Reference](#runner-script-reference)
10. [Output Structure](#output-structure)
11. [Resuming Runs](#resuming-runs)
12. [Experiment Groups (Current Config)](#experiment-groups-current-config)
13. [Extending the Pipeline](#extending-the-pipeline)

---

## Overview

The pipeline trains a model to denoise edge probability distributions over a graph. Given a set of candidate nodes (seed detections) and candidate edges between them, the model predicts which edges are part of the true underlying trace (neuron branch, vessel, etc.).

**Core idea:** At training time, a clean edge probability signal `p0` (ground truth labels) is corrupted via a cosine noise schedule over `T` steps, producing `p_t`. The model is trained to recover `p0` from `p_t`, conditioned on node patch features and a timestep embedding. At inference, the model is run iteratively from a max-noise initialization, progressively denoising toward a clean edge prediction.

---

## Repository Files

| File | Purpose |
|---|---|
| `train_gnn4.py` | Full training script — model definitions, loss functions, training loop, inference, plotting |
| `train_gnn4_params.txt` | Experiment parameter table — one row per experiment to launch |
| `train_gnn4_runner.sh` | SLURM batch launcher — reads the param file and dispatches one job array per row |

---

## Environment Setup

### Requirements

- Python 3.8+
- PyTorch (tested with CUDA 12.1)
- PyTorch Geometric or equivalent (for `.pt` graph files)
- `scipy` (for saving `.mat` validation outputs)
- `matplotlib`, `numpy`

### Conda environment

```bash
conda activate traceGNN
module load cuda/12.1
```

The runner script handles activation automatically. If your cluster uses a different module name, update the `CUDA_MODULE` variable in the runner or pass it at submission:

```bash
CUDA_MODULE=cuda/11.8 bash train_gnn4_runner.sh
```

---

## Data Format

Data is organized as a tree of run directories under a single `DATA_ROOT`:

```
DATA_ROOT/
  run1_full_pt/
    train/
      *.pt
    validation/
      *.pt
  run2_full_pt/
    train/
      *.pt
    validation/
      *.pt
  ...
```

The script collects all `.pt` files across any subdirectory starting with `RUN_PREFIX` (default `run1_full_pt`).

### `.pt` File Contents

Each `.pt` file is a PyTorch Geometric `Data` object with the following expected fields:

| Field | Shape | Description |
|---|---|---|
| `x` | `(N, F)` | Node features. First 3 columns are XYZ coordinates; remaining columns are patch descriptors |
| `edge_index` | `(2, E)` | Candidate edges as pairs of node indices |
| `edge_label` | `(E,)` | Ground truth binary edge labels (1 = true edge, 0 = false edge) |
| `max_noise_edge_index` | `(2, E_max)` | *(optional)* Edges present in the max-noise state, used to initialize inference |
| `R` | `(E,)` | *(optional)* Survival/selector values; fallback init if `max_noise_edge_index` absent |
| `true_positions` | `(N, 3)` | *(optional)* True node coordinates for visualization |
| `noisy_positions` | `(N, 3)` | *(optional)* Noisy node coordinates |

**Important:** `edge_label` values should be in `[0, 1]`. The training script binarizes them at threshold 0.5 internally.

---

## Model Architectures

Select via the `ARCH` column in the param file (or `--arch` CLI flag).

### `baseline`
Standard node Transformer + edge MLP head.
- Encodes all nodes with a `TransformerEncoder` (no spatial masking)
- Per-edge features: `[p_in, dist_z, sim_01, Δxyz, h_i, h_j]`
- Output: sigmoid edge probability

### `graphaware`
Same as baseline but with KNN-masked self-attention in the Transformer.
- Only attends to the `GRAPH_K` nearest neighbors per node (in coordinate space)
- Useful for large graphs where global attention is prohibitively expensive

### `edgecomp`
Adds an edge-level competition layer on top of the baseline node encoder.
- Projects per-edge tokens to a shared embedding space
- Runs a masked self-attention over edges that share a node (line-graph attention)
- Allows edges competing for the same node to suppress each other

### `edge_mlp`
Lightweight baseline — no Transformer, no node context.
- Scores edges from handcrafted features only: `[p_in, dist_z, sim_01, Δxyz]`
- Fastest to train; useful for ablation

### `edge_mlp_attn`
Lightweight with a single edge-level attention block.
- Same features as `edge_mlp` but with one `MultiheadAttention` layer over neighboring edges
- Good middle ground between speed and expressivity

### `temporal` / `combo`
Experimental wrappers adding cross-attention from edge tokens to a prior memory `{w, p_in}`. Not used in the current main experiments.

---

## Training Concepts

### Diffusion Schedule

A cosine noise schedule over `T` steps blends `p0` toward a prior `w`:

```
p_t = (1 - s_t) * p_{t-1} + s_t * w + ε
```

where `w` is a geometry/appearance prior (normalized blend of inverse distance and cosine patch similarity), `s_t` follows a cosine ramp from `s_min` to `s_max`, and `ε` is small Gaussian jitter.

### Curriculum

`start_min_t` to `end_min_t` controls the minimum diffusion step sampled during teacher-forced training. Early in training, the model only sees heavily corrupted inputs (large `t`). As training progresses, it also sees lightly corrupted inputs (small `t`), progressively forcing it to learn finer denoising.

### Free-Run Training

With probability `free_run_prob`, a batch skips teacher forcing and instead rolls out from the max-noise init for `free_run_steps` steps, computing loss at each step. This closes the train/test distribution gap and is critical for stable inference-time performance.

### Loss Functions

- **`focal`** (default): Focal loss with `focal_gamma=2.0`. Automatically sets `alpha` based on class imbalance. Preferred for sparse positive labels.
- **`bce`**: Weighted binary cross-entropy with explicit `pos_weight` from the positive/negative ratio.

### Loop Penalty

An auxiliary penalty added to the loss to discourage the model from predicting cycles. Three variants:

| `LOOP_TYPE` | Description |
|---|---|
| `none` | No loop penalty |
| `simple` | `tr(A³)/6` — triangle count on the soft adjacency matrix |
| `complex` | `tr(A³ (τI - A)⁻¹)` — resolvent-based penalty, geometrically discounts longer cycles |
| `nbt` | Non-backtracking (Hashimoto) operator penalty — penalizes closed walks while ignoring immediate edge reversals; generally preferred |

`LOOP_LAMBDA` controls the penalty weight. `LOOP_K` sets the number of additional powers in the series expansion. For `nbt`, `LOOP_NBT_WEIGHTING` controls how edge weights are propagated through the Hashimoto matrix (`sqrt` = geometric mean of adjacent edge weights; `child` = downstream edge weight only).

### Endpoint Penalty

Penalizes degree-1 nodes (dangling endpoints), which are artifacts of incomplete trace predictions. Four variants:

| `ENDPOINT_TYPE` | Description |
|---|---|
| `none` | No endpoint penalty |
| `bump` | Gaussian bump centered at degree=1; penalizes nodes with soft degree near 1 |
| `deg2` | Pushes active node degree toward 2 via `(deg - 2)²` |
| `quad` | L2 shrinkage on edge weights (`tr(A²)`) — negated to encourage sparsity |

---

## Running Experiments

### Local / interactive (single experiment)

```bash
python train_gnn4.py \
  --data-root /path/to/data \
  --run-prefix run1_full_pt \
  --arch graphaware \
  --graph-k 24 \
  --T 20 \
  --epochs 500 \
  --lr 0.001 \
  --batch-graphs 128 \
  --loss-type focal \
  --loop-penalty nbt \
  --loop-lambda 0.05 \
  --loop-K 6 \
  --loop-nbt-weighting sqrt \
  --loop-hutch-R 8 \
  --endpoint-penalty none \
  --endpoint-lambda 0.0 \
  --out-dir experiment_runs/my_run \
  --run-name my_experiment \
  --seed 1234
```

### SLURM (batch, from param file)

```bash
# CPU run (default)
PARAM_FILE=train_gnn4_params.txt bash train_gnn4_runner.sh

# GPU run
USE_GPU=1 GPU_PARTITION=gpu PARAM_FILE=train_gnn4_params.txt bash train_gnn4_runner.sh
```

This will:
1. Count the number of active (non-commented) rows in the param file
2. Create an output directory at `experiment_runs/<param_file_name>/`
3. Submit a SLURM job array with one task per experiment row

To run only a subset of experiments, comment out rows in the param file with `#`.

---

## Parameter File Reference

`train_gnn4_params.txt` is a whitespace-delimited table. Commented lines (starting with `#`) are skipped. Each active row defines one experiment run.

### Column Order

```
EXP_ID  DATA_ROOT  T  EPOCHS  LR  BATCH  LOSS  START  END  VAL_STEPS  VAL_DAMP  FREE  PIN_DROP  PIN_NOISE  FREEP  RUN_PREFIX  ARCH  GRAPH_K  LOOP_TYPE  LOOP_LAMBDA  LOOP_K  ENDPOINT_TYPE  ENDPOINT_LAMBDA  ENDPOINT_SIGMA  LOOP_TAU  LOOP_ESTIMATOR  LOOP_HUTCH_R  LOOP_NBT_WEIGHTING
```

### Column Descriptions

| Column | CLI Flag | Description |
|---|---|---|
| `EXP_ID` | `--run-name` | Unique experiment identifier; used as folder name |
| `DATA_ROOT` | `--data-root` | Root directory containing run subdirectories |
| `T` | `--T` | Diffusion steps (noise depth) |
| `EPOCHS` | `--epochs` | Total training epochs |
| `LR` | `--lr` | Learning rate |
| `BATCH` | `--batch-graphs` | Number of graphs per gradient update |
| `LOSS` | `--loss-type` | `focal` or `bce` |
| `START` | `--start-min-t` | Curriculum start (highest min timestep) |
| `END` | `--end-min-t` | Curriculum end (lowest min timestep, usually 1) |
| `VAL_STEPS` | `--val-infer-steps` | Inference steps during validation rollout |
| `VAL_DAMP` | `--val-damping` | Damping factor for reverse step: `p_next = (1-d)*p_in + d*p_hat` |
| `FREE` | `--free-run-steps` | Steps in free-run training mode |
| `PIN_DROP` | `--p-in-dropout` | Dropout rate on `p_in` (replaces dropped values with prior `w`) |
| `PIN_NOISE` | `--p-in-noise-std` | Gaussian noise std added to `p_in` during training |
| `FREEP` | `--free-run-prob` | Probability of using free-run instead of teacher forcing per batch |
| `RUN_PREFIX` | `--run-prefix` | Prefix of run subdirectories to collect data from |
| `ARCH` | `--arch` | Model architecture (see [Model Architectures](#model-architectures)) |
| `GRAPH_K` | `--graph-k` | KNN for graph-aware attention masking (only used if `ARCH=graphaware`) |
| `LOOP_TYPE` | `--loop-penalty` | Loop penalty type: `none`, `simple`, `complex`, `nbt` |
| `LOOP_LAMBDA` | `--loop-lambda` | Loop penalty weight |
| `LOOP_K` | `--loop-K` | Number of extra powers in loop penalty series |
| `ENDPOINT_TYPE` | `--endpoint-penalty` | Endpoint penalty type: `none`, `bump`, `deg2`, `quad` |
| `ENDPOINT_LAMBDA` | `--endpoint-lambda` | Endpoint penalty weight |
| `ENDPOINT_SIGMA` | `--endpoint-sigma` | Width of bump for `ENDPOINT_TYPE=bump` |
| `LOOP_TAU` | `--loop-tau` | τ in `(τI - A)⁻¹` for `LOOP_TYPE=complex`; must be > spectral radius of A |
| `LOOP_ESTIMATOR` | `--loop-complex-estimator` | `auto`, `exact`, or `hutch` (for `complex` penalty) |
| `LOOP_HUTCH_R` | `--loop-hutch-R` | Number of Hutchinson probe vectors |
| `LOOP_NBT_WEIGHTING` | `--loop-nbt-weighting` | `sqrt` or `child` (for `nbt` penalty) |

### Adding New Experiments

Append a new line to the param file (not starting with `#`). Do not change column order. Use `-` or `none` for unused penalty fields. Example:

```
my_new_run  /data/root  20  500  0.001  128  focal  8  1  20  0.6  2  0.70  0.30  0.50  run1_full_pt  graphaware  24  nbt  0.1  6  none  0.0  0.5  1.1  auto  8  sqrt
```

---

## Runner Script Reference (`train_gnn4_runner.sh`)

### Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PARAM_FILE` | `train_gnn4_params.txt` | Path to the experiment parameter file |
| `TOP_ROOT` | `experiment_runs` | Root directory for all output folders |
| `USE_GPU` | `0` | Set to `1` to request GPU resources |
| `GPUS` | `1` | Number of GPUs per job (if `USE_GPU=1`) |
| `GPU_PARTITION` | `gpu` | SLURM partition name for GPU jobs |
| `CUDA_MODULE` | `cuda/12.1` | CUDA module to load on the cluster |

### How the Runner Works

The script operates in two phases depending on whether `SLURM_ARRAY_TASK_ID` is set:

**Launcher phase** (no task ID set — first call):
1. Counts active rows in `PARAM_FILE`
2. Creates `TOP_ROOT/<param_file_stem>/` output root
3. Re-submits itself as a SLURM array (`--array=0-N`)

**Worker phase** (task ID set — inside the array):
1. Reads row `SLURM_ARRAY_TASK_ID` from `PARAM_FILE`
2. Parses all columns into shell variables
3. Assembles the Python argument list
4. Calls `srun python train_gnn4.py <args>`

Logs are written to `<OUT_ROOT>/logs/output_<EXP_ID>_<JOB_ID>.out` and `.err`.

---

## Output Structure

For each experiment, outputs are written to:

```
experiment_runs/
  train_gnn4_params/
    <EXP_ID>_<TIMESTAMP>/
      config_resolved.json          # Full config as JSON
      checkpoint_latest.pt          # Latest checkpoint (model + optimizer + history)
      best_model.pt                 # Best val BA checkpoint
      init_model_state.pt           # Random init weights (for epoch-0 baseline)
      baseline_epoch0_done.txt      # Flag file: epoch-0 baseline has been computed
      logs/
        output_<EXP_ID>_<JOB_ID>.out
        error_<EXP_ID>_<JOB_ID>.err
      figs/
        dashboard_epoch_NNN.png           # 9-panel training dashboard
        dashboard_combined_epoch_NNN.png  # 3-panel combined (train vs val) dashboard
        dashboard_last10_epoch_NNN.png    # 3-panel zoomed to last 10 epochs
        qual_epoch_NNN.png                # 2D MIP rollout visualization (pred vs GT)
        qual_epoch_trace_NNN.png          # 3D trace overlay (pred vs GT, top-down XY)
      validation_on_full_image_epochNNN/
        <aug1_filename>.mat               # Soft adjacency + edge probs for aug1 val files
```

### Checkpoint Format

Checkpoints are PyTorch `.pt` files with keys:
- `cfg`: Full config dict (from `dataclasses.asdict`)
- `model`: Model `state_dict`
- `optimizer`: Optimizer `state_dict`
- `epoch`: Last completed epoch
- `step`: Global gradient step count
- `history`: Dict of metric lists (`train_loss`, `val_loss`, `train_ba`, `val_ba`, etc.)

### `.mat` Validation Outputs

For every validation file with `aug1` (but not `aug10`, `aug11`, etc.) in its name, the script saves a `.mat` file containing:
- `A_prob`: `(N, N)` soft adjacency matrix
- `r`: `(N, 3)` node coordinates
- `p_edge`: `(E,)` per-edge predicted probabilities
- `edge_index`: `(2, E)` edge connectivity
- `pt_path`: Source file path string

These are intended for downstream MATLAB analysis of the full predicted graph.

---

## Resuming Runs

Resuming is automatic. If a run with the same `EXP_ID` prefix already exists in the output directory, the latest checkpoint is loaded and training continues from the last completed epoch.

To force a fresh run with the same name, either:
- Delete the existing run directory, or
- Change `EXP_ID` in the param file, or
- Set `--resume-if-found false` on the CLI

The epoch-0 baseline is also only computed once per run directory (guarded by `baseline_epoch0_done.txt`), so it will not be recomputed on resume.

---

## Experiment Groups (Current Config)

The current `train_gnn4_params.txt` organizes experiments into the following groups. All use the `21x21x7` patch data root and `T=20` unless noted.

### No-Loop Baseline Sweeps (`run_21_*`)

Architecture × diffusion depth grid with no loop penalty:

| T | Archs |
|---|---|
| 10, 20, 50, 100 | `baseline`, `graphaware` (K=24), `edgecomp` |

### Simple Loop Penalty (`simpleloop_*`)

Same grid as above with `LOOP_TYPE=simple`, `LOOP_LAMBDA=0.5`.

### Complex (Resolvent) Loop Penalty (`complexloop_*`)

`LOOP_TYPE=nbt`, λ ∈ {0.01, 0.05, 0.10}:
- `graphaware` K=24 / `edgecomp` / `baseline` with `sqrt` weighting
- `graphaware` K=24 / `edgecomp` with `child` weighting

### τ Sweep (`complexloop_tau*`)

Probes resolvent sensitivity by varying τ ∈ {1.05, 1.10, 1.20} at λ=0.05, `graphaware`.

### Hutchinson Estimator Sweep (`complexloop_hutchR*`)

Compares Hutchinson probe counts R ∈ {4, 16} vs the default auto/exact at λ=0.05, `baseline`.

### Endpoint Penalty Experiments (`EP_*`)

Combinations of loop and endpoint penalties:
- `bump`, `deg2`, `quad` endpoint types paired with `simple` and `nbt` loop penalties
- λ_loop=0.2, λ_endpoint=0.02

---

## Extending the Pipeline

### Adding a New Architecture

1. Define your model class in `train_gnn4.py` with the signature:
   ```python
   def forward(self, x_all, edge_index, p_in, t, dist_z, sim_01) -> torch.Tensor:
       # returns (E,) sigmoid probabilities
   ```
2. Add it to the `build_model()` and `model_factory()` functions under a new `cfg.arch` string.
3. Add the new string to the `--arch` argparse `choices` list.
4. Reference it in the param file via the `ARCH` column.

### Adding a New Loop Penalty

1. Implement a function `my_penalty(A, ...) -> torch.Tensor` operating on the soft adjacency matrix.
2. Add an `elif lp == "my_penalty":` branch in both the teacher-forced and free-run training paths inside `train()`.
3. Add the same branch to `eval_max_rollout_paths()` for consistent validation scoring.
4. Add the new name to the `--loop-penalty` argparse `choices` list.

### Changing the Dataset

The loader expects `.pt` files with `x`, `edge_index`, and optionally `edge_label`. Any graph dataset that produces PyTorch Geometric `Data` objects in this format is compatible. Update `DATA_ROOT` and `RUN_PREFIX` accordingly.