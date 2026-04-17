#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Diffusion — Residual Node Transformer + Edge Head
End-to-end training & validation with curriculum, checkpoints, and rich metrics/plots.

Assumptions:
- .pt graphs contain: x (N,F), edge_index (2,E), optional edge_label (E,)
- "max-noise state" may be present as one of:
  - max_noise_edge_index (2,Emax)
  - R (E_all,) survival/selector; edges with R<1 were used in prior code
  If none found, we fall back to label-free prior initialization for inference.
"""
from __future__ import annotations
import os, math, random, json, time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import argparse
from datetime import datetime

from mpl_toolkits.mplot3d import Axes3D

import glob

import re
try:
    import scipy.io as sio
except Exception:
    sio = None

def _csv_floats(s): 
    return [float(x) for x in s.split(',')] if isinstance(s, str) else s

def _csv_ints(s): 
    return [int(x) for x in s.split(',')] if isinstance(s, str) else s

def _bool(s):
    if isinstance(s, bool): return s
    s = s.lower()
    return s in ("1","true","t","yes","y","on")

@dataclass
class Config:
    # Repro
    seed: Optional[int] = None

    # Data
    data_root: str = "..."
    train_subdir: str = "train"
    val_subdir: str = "validation"
    run_prefix: str = "run1_full_pt"

    # Model
    # -------- Model selector & options --------
    arch: str = "baseline"   # ["baseline","graphaware","edgecomp","temporal","combo"]
    graph_k: int = 16        # KNN for graph-aware attention masking
    edge_mha_heads: int = 4  # edge-level self-attn heads
    edge_mha_layers: int = 1
    temporal_mha_heads: int = 2
    predict_residual: bool = False  # if True, output Δp and clamp(p_in + Δp)

    coord_dim: int = 3
    node_patch_emb_dim: int = 64
    d_model: int = 128
    nhead: int = 8
    nlayers: int = 2
    time_dim: int = 32
    edge_head_hidden: int = 128

    # Diffusion / corruption
    T: int = 10
    s_min: float = 0.05
    s_max: float = 0.7
    jitter_std: float = 0.05

    # Training
    epochs: int = 30
    lr: float = 1e-3
    batch_graphs: int = 1
    grad_clip: Optional[float] = 1.0
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha_base: float = 0.50

    # Curriculum
    use_curriculum: bool = True
    start_min_t: int = 8
    end_min_t: int = 1

    # Sampling
    train_frac: float = 1.0
    val_frac: float = 1.0

    # Logging / saving
    out_dir: str = "outputs_graph_diffusion_october"
    save_every_steps: int = 500
    infer_every_steps: int = 1000
    resume_if_found: bool = True
    ckpt_name: str = "checkpoint_latest.pt"

    # Threshold for metrics
    thresh: float = 0.5

    # NEW: p_in regularization + free-run
    p_in_dropout: float = 0.5
    p_in_noise_std: float = 0.20
    free_run_prob: float = 0.3
    free_run_steps: int = 2
    free_run_damping: float = 0.6
    
    # Inference/validation settings
    inference_mode: str = "max_noise"
    val_infer_steps: Optional[int] = None   # default: use T if None
    val_damping: float = 0.6

    # Train-eval from max-noise (subset so it’s quick)
    train_eval_frac: float = 0.25           # 25% of training loader
    train_eval_max_batches: Optional[int] = None
    
    # (keep these training scheduler knobs — you still use them)
    lr_sched_factor: float = 0.5
    lr_sched_patience: int = 5
    lr_sched_verbose: bool = True
    
    # Loop penalty
    loop_penalty: str = "none"     # ["none","simple","complex","nbt"]
    loop_lambda: float = 0.0       # weight of loop penalty
    loop_K: int = 6                # (complex) extra powers in truncated Neumann series
    loop_nbt_weighting: str = "sqrt"  # ["sqrt","child"]  sqrt(w_e*w_f) vs w_f
    
    # tau for (tau I - A)
    loop_tau: float = 1.1
    # estimator: "auto" picks exact for small N, Hutchinson for large N
    loop_complex_estimator: str = "auto"   # ["auto","exact","hutch"]
    loop_hutch_R: int = 8                  # # of probe vectors if using Hutchinson
    
    # Endpoint penalty (to mitigate gaps)
    endpoint_penalty: str = "none"    # ["none","bump","deg2","quad"]
    endpoint_lambda: float = 0.0      # overall weight
    endpoint_sigma: float = 0.5       # width of the degree≈1 bump
    endpoint_mask_gt: bool = True     # only apply on nodes touched by GT edges when teacher-forced


def make_cfg_from_cli() -> "Config":
    p = argparse.ArgumentParser(description="Graph Diffusion — training")

    # --- data & bookkeeping ---
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--train-subdir", type=str, default=None)
    p.add_argument("--val-subdir", type=str, default=None)
    p.add_argument("--run-prefix", type=str, default=None)

    # --- diffusion ---
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--s-min", type=float, default=None)
    p.add_argument("--s-max", type=float, default=None)
    p.add_argument("--jitter-std", type=float, default=None)

    # --- training ---
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-graphs", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    
    # --- architecture ---
    p.add_argument("--arch", type=str,
               choices=["baseline","graphaware","edgecomp","temporal","combo","edge_mlp","edge_mlp_attn"],
               default=None)
    p.add_argument("--graph-k", type=int, default=None)
    p.add_argument("--edge-mha-heads", type=int, default=None)
    p.add_argument("--edge-mha-layers", type=int, default=None)
    p.add_argument("--temporal-mha-heads", type=int, default=None)
    p.add_argument("--predict-residual", type=_bool, default=None)


    # --- loss ---
    p.add_argument("--loss-type", type=str, choices=["bce", "focal"], default=None)
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--focal-alpha-base", type=float, default=None)

    # --- curriculum ---
    p.add_argument("--use-curriculum", type=_bool, default=None)
    p.add_argument("--start-min-t", type=int, default=None)
    p.add_argument("--end-min-t", type=int, default=None)

    p.add_argument("--inference-mode", type=str, choices=["max_noise", "prior_only"], default=None)

    # --- thresholds & scoring ---
    p.add_argument("--thresh", type=float, default=None)

    p.add_argument("--tangent-k", type=int, default=None)

    # --- p_in regularization + free-run ---
    p.add_argument("--p-in-dropout", type=float, default=None)
    p.add_argument("--p-in-noise-std", type=float, default=None)
    p.add_argument("--free-run-prob", type=float, default=None)
    p.add_argument("--free-run-steps", type=int, default=None)
    p.add_argument("--free-run-damping", type=float, default=None)


    # --- io & reproducibility ---
    p.add_argument("--save-every-steps", type=int, default=None)
    p.add_argument("--infer-every-steps", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None, help="optional suffix; a timestamp is appended")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--resume-if-found", type=_bool, default=None)
    
    p.add_argument("--val-infer-steps", type=int, default=None)
    p.add_argument("--val-damping", type=float, default=None)
    p.add_argument("--train-eval-frac", type=float, default=None)
    p.add_argument("--train-eval-max-batches", type=int, default=None)
    
    # --- loop penalty ---
    p.add_argument("--loop-penalty", type=str, choices=["none","simple","complex","nbt"], default=None)
    p.add_argument("--loop-nbt-weighting", type=str, choices=["sqrt","child"], default=None)
    p.add_argument("--loop-lambda", type=float, default=None)
    p.add_argument("--loop-K", type=int, default=None)
    
    # --- endpoint penalty ---
    p.add_argument("--endpoint-penalty", type=str, choices=["none","bump","deg2","quad"], default=None)
    p.add_argument("--endpoint-lambda", type=float, default=None)
    p.add_argument("--endpoint-sigma", type=float, default=None)
    p.add_argument("--endpoint-mask-gt", type=_bool, default=None)
    
    p.add_argument("--loop-tau", type=float, default=None)
    p.add_argument("--loop-complex-estimator", type=str, choices=["auto","exact","hutch"], default=None)
    p.add_argument("--loop-hutch-R", type=int, default=None)
    
    args = p.parse_args()
    cfg = Config()  # defaults

    # carry explicit resume flag if provided
    if args.resume_if_found is not None:
        cfg.resume_if_found = bool(args.resume_if_found)

    base_out = args.out_dir or cfg.out_dir
    run_name = args.run_name

    # Smart run directory resolution
    if run_name:
        # candidates like OUT_DIR/run_name_20241012_153012
        pattern = os.path.join(base_out, f"{run_name}_*")
        matches = sorted(glob.glob(pattern))
        if cfg.resume_if_found and matches:
            # pick newest by lexicographic ts suffix
            out_dir = matches[-1]
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(base_out, f"{run_name}_{ts}")
    else:
        # no run_name ⇒ use base_out directly
        out_dir = base_out

    cfg.out_dir = out_dir

    # Apply the rest of args generically (skip specials)
    for k, v in vars(args).items():
        if k in ("out_dir", "run_name", "resume_if_found") or v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    ensure_dir(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "config_resolved.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return cfg


# ------------------------ Utilities -----------------------------

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def minmax01(x: torch.Tensor) -> torch.Tensor:
    xmin = torch.min(x); xmax = torch.max(x)
    if (xmax - xmin) < 1e-8:
        return torch.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean(); s = x.std()
    return (x - m) / (s + eps)
    
def build_train_init(sample, w):
    # reuse your inference init
    E = sample["edge_index"].size(1)
    p0_init = build_init_from_max_noise(sample, E)
    return p0_init if p0_init is not None else w
    
def build_soft_adjacency(pred_sig: torch.Tensor, edge_index: torch.Tensor, N: Optional[int] = None) -> torch.Tensor:
    """
    pred_sig: (E,) sigmoid outputs in [0,1]
    edge_index: (2,E)
    Returns A in R^{N x N}, symmetric, zero diagonal, weighted by pred_sig.
    """
    i, j = edge_index
    if N is None:
        N = int(max(int(i.max()), int(j.max())) + 1)
    A = torch.zeros((N, N), dtype=pred_sig.dtype, device=pred_sig.device)
    A.index_put_((i, j), pred_sig, accumulate=True)
    A.index_put_((j, i), pred_sig, accumulate=True)  # assume undirected
    A.fill_diagonal_(0.0)
    # clip just in case of duplicated edges accumulating >1
    return A.clamp(0.0, 1.0)


def triangle_penalty(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Weighted triangle mass: trace(A^3)/6 for undirected graphs.
    Normalized by (N + E) to keep scale tame across sizes.
    """
    N = A.size(0)
    A2 = A @ A
    t3 = torch.trace(A2 @ A) / 6.0
    # scale: rough size proxy (prevents tiny graphs from dominating)
    E_est = A.sum() / 2.0
    denom = (N + E_est + eps)
    return t3 / denom

'''
def triangle_penalty(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Modified loop penalty using:
        num = tr(A^3 - (A^2 / 3))

    A: soft adjacency in [0,1], symmetric, zero diagonal.
    We normalize by (N + E_est) to keep scale comparable across graph sizes.
    """
    N = A.size(0)

    # powers
    A2 = A @ A          # A^2
    A3 = A2 @ A         # A^3

    # new numerator: tr(A^3 - A^2 / 3)
    num = torch.trace(A3 - A2 / 3.0)

    # same scale proxy as before
    E_est = A.sum() / 2.0
    denom = (N + E_est + eps)

    return num / denom
'''

def neumann_cycle_penalty(A: torch.Tensor, K: int = 6, eps: float = 1e-8) -> torch.Tensor:
    """
    Approximates A^3 (I - A)^{-1} via truncated Neumann series:
      A^3 * sum_{m=0..K} A^m = sum_{k=3..3+K} A^k
    Penalize the diagonal mass (closed walks) across lengths >=3.
    """
    N = A.size(0)
    # optional stabilizer: scale if spectral radius might be >1
    # (detach scale to avoid backprop into it)
    with torch.no_grad():
        s = A.max().clamp(min=eps)  # cheap upper bound proxy
        scale = (0.9 / s).clamp(max=1.0)  # <=1
    A_scaled = A * scale

    Ak = A_scaled @ A_scaled @ A_scaled  # A^3
    total = torch.trace(Ak)
    for _ in range(K):
        Ak = Ak @ A_scaled               # next power
        total = total + torch.trace(Ak)

    E_est = A.sum() / 2.0
    denom = (N + E_est + eps)
    return total / denom
    
def resolvent_cycle_penalty(
    A: torch.Tensor,
    tau: float = 1.1,
    estimator: str = "auto",
    R: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize closed walks of length >=3 with geometric discount via:
        tr( A^3 (tau I - A)^(-1) )

    Computation strategy:
      - EXACT:    solve (tau I - A) Y = A^3  -> tr(Y)
      - HUTCH:    tr(M) ≈ (1/R) Σ z^T M z with M = A^3 (tau I - A)^(-1)
                  For each probe z: y solves (tau I - A) y = A^3 z, then z^T y.

    Notes:
      * Uses linear solves, NOT explicit matrix inverse (more stable and autograd-friendly).
      * Adds tiny diagonal eps to guard against near-singularity.
    """
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be square"
    N = A.size(0)
    device, dtype = A.device, A.dtype

    # build operator C = tau I - A
    I = torch.eye(N, device=device, dtype=dtype)
    C = tau * I - A
    C = C + eps * I  # tiny Tikhonov to improve conditioning

    # A^3
    A2 = A @ A
    A3 = A2 @ A

    # choose estimator
    if estimator == "auto":
        # rough crossover; exact is O(N^3). You can tune this.
        estimator = "exact" if N <= 256 else "hutch"

    if estimator == "exact":
        # Solve (tau I - A) Y = A^3  for all RHS columns at once
        # -> one batched solve, then take trace(Y)
        # torch.linalg.solve handles autograd
        Y = torch.linalg.solve(C, A3)
        val = torch.trace(Y)
    elif estimator == "hutch":
        # Hutchinson: tr(M) ≈ (1/R) Σ z^T M z, M = A^3 C^{-1}
        # Draw Rademacher probes
        acc = A.new_zeros(())
        for _ in range(int(R)):
            z = torch.empty((N,1), device=device, dtype=dtype).bernoulli_().mul_(2).add_(-1)  # ±1
            A3z = A3 @ z
            y   = torch.linalg.solve(C, A3z)  # C y = A^3 z
            acc = acc + (z.t() @ y).squeeze()
        val = acc / float(R)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # normalization similar to your other penalties
    with torch.no_grad():
        E_est = A.sum() / 2.0
        denom = (N + E_est).clamp_min(1.0)

    return (val / denom)
    
# ---- End point penalty helper functions --------
    
def soft_degree(A: torch.Tensor) -> torch.Tensor:
    # A is symmetric, zero-diagonal
    return A.sum(dim=1)  # (N,)

def endpoint_bump_penalty(
    A: torch.Tensor, sigma: float = 0.5, node_mask: Optional[torch.Tensor] = None, eps: float = 1e-8
) -> torch.Tensor:
    """
    Penalize nodes with soft degree near 1 using a Gaussian bump centered at 1.
      L = mean( exp(-0.5 * ((deg-1)/sigma)^2) ) over masked nodes
    Small if deg≈0 or deg≥2; large near deg≈1.
    """
    deg = soft_degree(A)
    bump = torch.exp(-0.5 * ((deg - 1.0) / max(sigma, 1e-6))**2)
    if node_mask is not None:
        m = node_mask.float()
        return (bump * m).sum() / (m.sum() + eps)
    return bump.mean()

def degree_to_two_penalty(
    A: torch.Tensor, node_mask: Optional[torch.Tensor] = None, eps: float = 1e-8
) -> torch.Tensor:
    """
    Softer alternative: push active-node degree toward 2:
      L = mean( (deg>0) * (deg-2)^2 ), optionally masked.
    """
    deg = soft_degree(A)
    active = (deg > 0).float()
    term = ((deg - 2.0)**2) * active
    if node_mask is not None:
        m = node_mask.float() * active
        return (term * m).sum() / (m.sum() + eps)
    return term.mean()
    
def gt_node_mask_from_edges(edge_index: torch.Tensor, y_edge_bin: torch.Tensor, N: int) -> torch.Tensor:
    """
    y_edge_bin: (E,) in {0,1} ; returns (N,) bool marking nodes touched by any GT-positive edge.
    """
    i, j = edge_index
    pos = (y_edge_bin > 0)
    mask = torch.zeros(N, dtype=torch.bool, device=edge_index.device)
    mask.index_fill_(0, i[pos], True)
    mask.index_fill_(0, j[pos], True)
    return mask
    
def _build_hashimoto_nonbacktracking(
    edge_index: torch.Tensor,
    p_edge: torch.Tensor,
    N: Optional[int] = None,
    weighting: str = "sqrt",
) -> torch.Tensor:
    """
    Build sparse non-backtracking (Hashimoto) operator B for an undirected, weighted graph.
    Each undirected edge {u,v} becomes two directed edges u->v and v->u.
    B_{(u->v),(x->y)} = weight if (v==x) and (y!=u), else 0.
    
    weighting:
      - "sqrt":  weight = sqrt(w_{u->v} * w_{x->y})  (default; smooth & symmetric)
      - "child": weight = w_{x->y}                   (passes the next-edge weight)
      
    Returns: B as a torch.sparse_coo_tensor of shape (2E, 2E).
    """
    device = edge_index.device
    dtype  = p_edge.dtype
    i, j = edge_index
    E = i.numel()
    if E == 0:
        return torch.sparse_coo_tensor(torch.empty((2,0), dtype=torch.long, device=device),
                                       torch.empty((0,), dtype=dtype, device=device),
                                       size=(0,0))

    # build directed-edge lists
    src_dir = torch.cat([i, j], dim=0)  # (2E,)
    dst_dir = torch.cat([j, i], dim=0)  # (2E,)
    w_dir   = torch.cat([p_edge, p_edge], dim=0)  # (2E,)
    idx_undirected = torch.arange(E, device=device, dtype=torch.long)
    idx_dir2und = torch.cat([idx_undirected, idx_undirected], dim=0)  # map dir-edge -> undirected id

    # For each node v, connect all incoming dir-edges (*->v) to outgoing dir-edges (v->*)
    # but disallow immediate reversal: next_dst != prev_src
    N_nodes = int(max(int(src_dir.max()), int(dst_dir.max())) + 1) if N is None else int(N)

    # incidence lists per node (as indices in 2E space)
    # In practice, building these via Python loops over nodes is fast for medium graphs.
    incoming = [[] for _ in range(N_nodes)]
    outgoing = [[] for _ in range(N_nodes)]
    # fill
    for idx in range(src_dir.numel()):
        u = int(src_dir[idx].item()); v = int(dst_dir[idx].item())
        outgoing[u].append(idx)  # u->v
        incoming[v].append(idx)  # u->v ends at v

    rows = []
    cols = []
    vals = []

    # assemble B by nodes
    for v in range(N_nodes):
        ins  = incoming[v]
        outs = outgoing[v]
        if not ins or not outs:
            continue

        # tensors for vectorized masking
        ins_t  = torch.as_tensor(ins, device=device, dtype=torch.long)
        outs_t = torch.as_tensor(outs, device=device, dtype=torch.long)

        prev_src = src_dir.index_select(0, ins_t)   # (I,)
        next_dst = dst_dir.index_select(0, outs_t)  # (O,)

        # all pairs (ins x outs)
        I = ins_t.numel(); O = outs_t.numel()
        if I == 0 or O == 0:
            continue
        ii = ins_t.view(I, 1).expand(I, O).reshape(-1)   # row indices in 2E space
        oo = outs_t.view(1, O).expand(I, O).reshape(-1)  # col indices in 2E space

        # disallow immediate reversal: next_dst != prev_src
        ps = prev_src.view(I, 1).expand(I, O).reshape(-1)
        nd = next_dst.view(1, O).expand(I, O).reshape(-1)
        keep = (nd != ps)

        if keep.any():
            ii = ii[keep]; oo = oo[keep]
            if weighting == "child":
                w_next = w_dir.index_select(0, oo)
                w_val = w_next
            else:
                w_prev = w_dir.index_select(0, ii)
                w_next = w_dir.index_select(0, oo)
                w_val = torch.sqrt((w_prev * w_next).clamp_min(0.0))
            rows.append(ii); cols.append(oo); vals.append(w_val)

    if len(rows) == 0:
        # no connections -> zero operator
        return torch.sparse_coo_tensor(torch.empty((2,0), dtype=torch.long, device=device),
                                       torch.empty((0,), dtype=dtype, device=device),
                                       size=(2*E, 2*E))
    rows = torch.cat(rows, dim=0)
    cols = torch.cat(cols, dim=0)
    vals = torch.cat(vals, dim=0)

    B = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0),
        vals, size=(2*E, 2*E), device=device, dtype=dtype
    ).coalesce()
    return B


def _sparse_mm(B: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Sparse-dense matmul that accepts (2E,2E) @ (2E,k) or (2E,) -> (2E,k)/(2E,)."""
    if X.dim() == 1:
        return torch.sparse.mm(B, X.view(-1,1)).view(-1)
    return torch.sparse.mm(B, X)

def _row_abs_sums_sparse(B: torch.Tensor) -> torch.Tensor:
    """Return L1 row sums of sparse matrix B (coalesced)."""
    Bc = B.coalesce()
    idx = Bc.indices()          # (2, nnz)
    vals = Bc.values().abs()    # (nnz,)
    nrows = Bc.size(0)
    rs = torch.zeros((nrows,), device=vals.device, dtype=vals.dtype)
    rs.index_add_(0, idx[0], vals)
    return rs

def nonbacktracking_loop_penalty(
    edge_index: torch.Tensor,
    p_edge: torch.Tensor,
    N_nodes: int,
    K: int = 6,
    R: int = 8,
    weighting: str = "sqrt",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Hutchinson estimate of tr( B^3 * sum_{m=0..K} B^{m} ), with B the non-backtracking (Hashimoto) operator.
    This counts **all** non-backtracking closed walks of length >=3 (odd + even), while still
    ignoring immediate edge reversals.

    Args:
      edge_index: (2,E) long
      p_edge:     (E,) float in [0,1] (soft edge predictions)
      N_nodes:    number of nodes, for normalization
      K:          number of extra powers beyond 3 (>=0)  => total powers 3..(3+K)
      R:          Hutchinson probe count
      weighting:  "sqrt" or "child" for B weights (see builder)
    """
    device, dtype = edge_index.device, p_edge.dtype
    E = p_edge.numel()
    if E == 0:
        return torch.zeros((), device=device, dtype=dtype)

    # Build B (2E x 2E) sparse
    B = _build_hashimoto_nonbacktracking(edge_index, p_edge, N=N_nodes, weighting=weighting).coalesce()
    D = B.size(0)  # 2E

    if D == 0 or B._nnz() == 0:
        return torch.zeros((), device=device, dtype=dtype)

    # --- Stability: scale B so spectral radius < 1 (cheap bound via max row L1 norm) ---
    row_l1 = _row_abs_sums_sparse(B)
    max_l1 = torch.maximum(row_l1.max(), torch.tensor(1.0, device=device, dtype=dtype))
    scale = (0.9 / max_l1).clamp(max=1.0)  # keep <=1 so we never inflate
    B = torch.sparse_coo_tensor(B.indices(), B.values() * scale, size=B.size(), device=device, dtype=dtype).coalesce()

    acc = p_edge.new_zeros(())
    for _ in range(int(R)):
        # Rademacher probe
        z = torch.empty((D,), device=device, dtype=dtype).bernoulli_().mul_(2).add_(-1)

        # y3 = B^3 z
        y1 = _sparse_mm(B, z)
        y2 = _sparse_mm(B, y1)
        yk = _sparse_mm(B, y2)  # this is B^3 z

        # accumulate z^T (B^3 + B^4 + ... + B^{3+K}) z
        term = (z * yk).sum()
        for _m in range(int(K)):
            yk = _sparse_mm(B, yk)  # multiply by B once each time -> includes ALL lengths
            term = term + (z * yk).sum()

        acc = acc + term

    val = acc / float(R)

    # normalization (similar scale as other penalties)
    with torch.no_grad():
        E_est = p_edge.sum()  # soft total edge mass
        denom = (N_nodes + 0.5 * E_est).clamp_min(1.0)
    return val / denom

    
def quadratic_edge_penalty(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L2 shrinkage of soft edge weights:
        tr(A^2)  ==  ||A||_F^2   (for symmetric A)
    For undirected graphs with zero diagonal, this equals 2 * sum_e w_e^2.
    We normalize by (N + E_est) to keep scale comparable across sizes.
    """
    N = A.size(0)
    val = torch.trace(A @ A)  # tr(A^2) == ||A||_F^2
    E_est = A.sum() / 2.0
    denom = (N + E_est + eps)
    return val / denom

# ---------------------- Data Loading ----------------------------

'''
def load_pt_file(file_path: str) -> Any:
    data = torch.load(file_path, weights_only=False)
    # normalize dtypes
    if hasattr(data, 'x'): data.x = data.x.to(torch.float32)
    if hasattr(data, 'edge_index'): data.edge_index = data.edge_index.to(torch.long)
    if hasattr(data, 'edge_label') and data.edge_label is not None:
        data.edge_label = data.edge_label.to(torch.float32).view(-1)
    data.pt_path = file_path
    return data
'''
    
def load_pt_file(file_path: str) -> Any:
    # compatible with both old/new torch.load signatures
    try:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch: no 'weights_only' kwarg
        data = torch.load(file_path, map_location="cpu")

    # normalize dtypes
    if hasattr(data, 'x'): data.x = data.x.to(torch.float32)
    if hasattr(data, 'edge_index'): data.edge_index = data.edge_index.to(torch.long)
    if hasattr(data, 'edge_label') and data.edge_label is not None:
        data.edge_label = data.edge_label.to(torch.float32).view(-1)
    data.pt_path = file_path
    return data


def list_pt_files_across_runs(data_root: str, subfolder_name: str, run_prefix: str) -> List[str]:
    paths = []
    for d in os.listdir(data_root):
        run_dir = os.path.join(data_root, d)
        if not d.startswith(run_prefix): continue
        if not os.path.isdir(run_dir): continue
        target = os.path.join(run_dir, subfolder_name)
        if os.path.exists(target):
            for root, _, files in os.walk(target):
                for f in files:
                    if f.endswith(".pt"):
                        paths.append(os.path.join(root, f))
    paths.sort()
    return paths

class GraphPathsDataset(Dataset):
    def __init__(self, pt_paths: List[str], device: torch.device):
        self.pt_paths = pt_paths
        self.device = device
    def __len__(self) -> int: return len(self.pt_paths)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = load_pt_file(self.pt_paths[idx])
        out = {
            "x_all": data.x.to(self.device),
            "edge_index": data.edge_index.to(self.device),
            "p0": (data.edge_label.to(self.device) if hasattr(data, "edge_label") and data.edge_label is not None else None),
            "path": getattr(data, "pt_path", self.pt_paths[idx]),
        }
        # Optional max-noise carriers
        if hasattr(data, "max_noise_edge_index"):
            out["max_noise_edge_index"] = data.max_noise_edge_index.to(self.device)
        if hasattr(data, "R"):
            out["R"] = data.R.to(self.device)
        # optional positions
        if hasattr(data, "true_positions"): out["true_positions"] = data.true_positions.to(self.device)
        if hasattr(data, "noisy_positions"): out["noisy_positions"] = data.noisy_positions.to(self.device)
        return out

def collate_list(batch):
    return batch  # keep a list; we iterate manually
    
def knn_mask_from_coords(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return an (N,N) attention mask with 0 for allowed and -inf for disallowed pairs,
    keeping self + k nearest neighbors per node. Suitable for nn.TransformerEncoder src_mask.
    """
    with torch.no_grad():
        N = coords.size(0)
        if N == 0:
            return torch.zeros((0,0), device=coords.device)
        D = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0), p=2).squeeze(0)  # (N,N)
        kn = min(k+1, N)  # include self
        _, idx = torch.topk(-D, k=kn, dim=1)  # most negative distance = closest
        allow = torch.zeros((N, N), dtype=torch.bool, device=coords.device)
        row = torch.arange(N, device=coords.device).unsqueeze(1).expand(N, kn)
        allow[row, idx] = True
        mask = torch.zeros((N, N), device=coords.device)
        mask[~allow] = float("-inf")
        return mask

def edge_share_node_mask(edge_index: torch.Tensor) -> torch.Tensor:
    """
    For E edges, returns (E,E) mask with 0 for edges that share a node, -inf otherwise.
    """
    i, j = edge_index
    E = i.numel()
    with torch.no_grad():
        # Encode endpoints as tuples; build incidence maps
        # Faster approach: compare endpoints via broadcasting
        share = (i.view(E,1) == i.view(1,E)) | (i.view(E,1) == j.view(1,E)) | \
                (j.view(E,1) == i.view(1,E)) | (j.view(E,1) == j.view(1,E))
        mask = torch.zeros((E, E), device=edge_index.device)
        mask[~share] = float("-inf")
        return mask


# --------------------- Edge Features & Prior --------------------

def pairwise_distances_from_coords(edge_index: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    i, j = edge_index
    diff = coords[i] - coords[j]
    return torch.linalg.norm(diff, dim=1)

def cosine_patch_similarity(edge_index: torch.Tensor, feats: torch.Tensor, start_idx: int = 3) -> torch.Tensor:
    i, j = edge_index
    A = feats[i, start_idx:]
    B = feats[j, start_idx:]
    A = F.normalize(A, dim=1); B = F.normalize(B, dim=1)
    return (A * B).sum(dim=1)  # [-1,1]

@torch.no_grad()
def compute_prior_and_edgefeats(sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_all = sample["x_all"]; ei = sample["edge_index"]
    coords = x_all[:, :3]
    d = pairwise_distances_from_coords(ei, coords)       # raw distance
    s = cosine_patch_similarity(ei, x_all)               # cosine [-1,1]

    # Prior w: geom/app blend
    geom = 1.0 - minmax01(d)                # closer => larger
    app  = minmax01((s + 1.0) / 2.0)        # map to [0,1]
    w = minmax01(0.6*geom + 0.4*app)

    # Normalize edge features for the head
    d_z = zscore(d)
    s_01 = (s + 1.0) / 2.0                  # keep in [0,1]
    return w, d_z, s_01

# --------------------- Schedules & Corruption -------------------

def cosine_schedule(T: int, s_min: float = 0.05, s_max: float = 0.4, device=device) -> torch.Tensor:
    t = torch.linspace(0, math.pi, steps=T, device=device)
    raw = 0.5*(1 - torch.cos(t))  # [0,1]
    return s_min + (s_max - s_min) * raw

def corrupt_once(p_prev: torch.Tensor, w: torch.Tensor, s_t: float, jitter_std: float) -> torch.Tensor:
    eps = torch.randn_like(p_prev) * jitter_std
    p_t = (1.0 - s_t) * p_prev + s_t * w + eps
    return p_t.clamp(0.0, 1.0)

def build_p_t_from_sched(p0: torch.Tensor, w: torch.Tensor, sched: torch.Tensor, t: int, jitter_std: float) -> torch.Tensor:
    p = p0
    for k in range(int(t)):
        p = corrupt_once(p, w, float(sched[k].item()), jitter_std)
    return p

def make_inference_schedule(T: int, steps: int = 4, mode: str = "cosine") -> List[int]:
    if steps <= 1: return [T]
    if mode == "linear":
        ts = torch.linspace(T, 1, steps)
    else:
        xs = torch.linspace(0, math.pi, steps)
        frac = 0.5*(1 - torch.cos(xs))
        ts = T - (T-1) * frac
    ts = ts.round().long().tolist()
    ts = sorted(set(ts), reverse=True)
    if ts[-1] != 1: ts.append(1)
    return ts

# --------------------------- Model ------------------------------

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.register_buffer('freqs', torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), steps=dim//2)))
    def forward(self, t_scalar: float):
        t = torch.tensor(float(t_scalar), device=self.freqs.device).view(1,1)
        angles = t * self.freqs.view(1,-1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.size(1) < self.dim:
            emb = F.pad(emb, (0, self.dim-emb.size(1)))
        return emb.squeeze(0)  # (dim,)

class NodePatchEncoder(nn.Module):
    def __init__(self, patch_dim: int, out_dim: int = 64):
        super().__init__()
        #self.net = nn.Sequential(
        #    nn.Linear(patch_dim, 256), nn.ReLU(),
        #    nn.Linear(256, out_dim), nn.ReLU(),
        #)
        self.net = nn.Sequential(
            nn.Linear(patch_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, out_dim), nn.ReLU(), nn.Dropout(0.2),
        )
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.net(patches)

class NodeTransformerEdgeHead(nn.Module):
    """
    - Node tokens: z-scored coords + (optional) patch MLP + time embedding
    - TransformerEncoder over nodes
    - Edge head takes: p_in, dist_z, sim_01, Δxyz (3), and hi,hj
    """
    def __init__(self, coord_dim=3, patch_dim=0, node_emb_dim=64, d_model=128, nhead=8, nlayers=2,
                 time_dim=32, edge_head_hidden=128):
        super().__init__()
        self.time_tok = SinusoidalTimeEmbed(time_dim)
        self.use_patch = patch_dim > 0
        self.patch_enc = NodePatchEncoder(patch_dim, node_emb_dim) if self.use_patch else None

        node_in = coord_dim + (node_emb_dim if self.use_patch else 0) + time_dim
        self.node_proj = nn.Linear(node_in, d_model)

        #enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        edge_in = (2*d_model) + 1 + 1 + 1 + 3  # p_in + dist_z + sim_01 + delta_xyz(3) + hi + hj
        '''
        self.edge_head = nn.Sequential(
            nn.Linear(edge_in, edge_head_hidden), nn.ReLU(),
            nn.Linear(edge_head_hidden, edge_head_hidden), nn.ReLU(),
            nn.Linear(edge_head_hidden, 1), nn.Sigmoid()
        )
        '''
        self.edge_head = nn.Sequential(
            nn.Linear(edge_in, edge_head_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(edge_head_hidden, edge_head_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(edge_head_hidden, 1), nn.Sigmoid()
        )

    def forward(self, x_all: torch.Tensor, edge_index: torch.Tensor, p_in: torch.Tensor, t: int,
                dist_z: torch.Tensor, sim_01: torch.Tensor) -> torch.Tensor:
        N = x_all.size(0)
        coords = x_all[:, :3]
        coords_z = (coords - coords.mean(0, keepdim=True)) / (coords.std(0, keepdim=True) + 1e-6)

        feats = [coords_z]
        if self.use_patch:
            patches = x_all[:, 3:]
            node_patch = self.patch_enc(patches)
            feats.append(node_patch)

        emb_t = self.time_tok(t).to(x_all.device).view(1,-1).repeat(N,1)
        feats.append(emb_t)

        node_tokens = torch.cat(feats, dim=1)
        node_h = self.node_proj(node_tokens)
        node_h = self.encoder(node_h.unsqueeze(0)).squeeze(0)  # (N, d_model)

        i, j = edge_index
        hi, hj = node_h[i], node_h[j]
        delta = coords_z[i] - coords_z[j]

        edge_feats = torch.cat([
            p_in.view(-1,1),
            dist_z.view(-1,1),
            sim_01.view(-1,1),
            delta,
            hi, hj
        ], dim=1)
        return self.edge_head(edge_feats).view(-1)
        
    def encode_nodes(self, x_all: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = x_all[:, :3]
        coords_z = (coords - coords.mean(0, keepdim=True)) / (coords.std(0, keepdim=True) + 1e-6)
        feats = [coords_z]
        if self.use_patch:
            feats.append(self.patch_enc(x_all[:, 3:]))
        emb_t = self.time_tok(t).to(x_all.device).view(1,-1).repeat(x_all.size(0),1)
        feats.append(emb_t)
        node_tokens = torch.cat(feats, dim=1)
        node_h = self.node_proj(node_tokens)
        node_h = self.encoder(node_h.unsqueeze(0)).squeeze(0)
        return node_h, coords_z

class GraphAwareNodeTransformerEdgeHead(NodeTransformerEdgeHead):
    """
    Same as baseline, but masks node self-attention to KNN using coords (graph-aware).
    """
    def __init__(self, *args, graph_k: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_k = graph_k

    def forward(self, x_all, edge_index, p_in, t, dist_z, sim_01):
        N = x_all.size(0)
        coords = x_all[:, :3]
        coords_z = (coords - coords.mean(0, keepdim=True)) / (coords.std(0, keepdim=True) + 1e-6)

        feats = [coords_z]
        if self.use_patch:
            patches = x_all[:, 3:]
            feats.append(self.patch_enc(patches))
        emb_t = self.time_tok(t).to(x_all.device).view(1,-1).repeat(N,1)
        feats.append(emb_t)

        node_tokens = torch.cat(feats, dim=1)
        node_h = self.node_proj(node_tokens)

        # Build KNN attention mask (seq,seq)
        attn_mask = knn_mask_from_coords(coords_z, self.graph_k)
        node_h = self.encoder(node_h.unsqueeze(0), mask=attn_mask).squeeze(0)

        i, j = edge_index
        hi, hj = node_h[i], node_h[j]
        delta = coords_z[i] - coords_z[j]

        edge_feats = torch.cat([
            p_in.view(-1,1), dist_z.view(-1,1), sim_01.view(-1,1),
            delta, hi, hj
        ], dim=1)
        out = self.edge_head(edge_feats).view(-1)
        return out
    
    def encode_nodes(self, x_all, t):
        coords = x_all[:, :3]
        coords_z = (coords - coords.mean(0, keepdim=True)) / (coords.std(0, keepdim=True) + 1e-6)
        feats = [coords_z]
        if self.use_patch:
            feats.append(self.patch_enc(x_all[:, 3:]))
        emb_t = self.time_tok(t).to(x_all.device).view(1,-1).repeat(x_all.size(0),1)
        feats.append(emb_t)
        node_tokens = torch.cat(feats, dim=1)
        node_h = self.node_proj(node_tokens)
        mask = knn_mask_from_coords(coords_z, self.graph_k)
        node_h = self.encoder(node_h.unsqueeze(0), mask=mask).squeeze(0)
        return node_h, coords_z


class EdgeCompetitionWrapper(nn.Module):
    def __init__(self, base_node_encoder: NodeTransformerEdgeHead,
                 d_edge: int = 256, heads: int = 4, layers: int = 1,
                 predict_residual: bool = False):
        super().__init__()
        self.base = base_node_encoder
        self.predict_residual = predict_residual

        in_edge_tok = 5 + 2 * self.base.node_proj.out_features  # 1+1+3 + 2*d_model
        self.edge_proj = nn.Linear(in_edge_tok, d_edge)

        self.edge_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_edge, num_heads=heads, batch_first=True)
            for _ in range(layers)
        ])
        self.edge_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_edge, d_edge*2), nn.ReLU(), nn.Linear(d_edge*2, d_edge))
            for _ in range(layers)
        ])
        self.edge_head = nn.Sequential(
            nn.Linear(d_edge + 1, d_edge), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_edge, d_edge//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_edge//2, 1), nn.Sigmoid()
        )

    def _build_edge_tokens(self, x_all, edge_index, p_in, t, dist_z, sim_01):
        # reuse the base to get node_h & coords_z, but intercept pre-head features
        N = x_all.size(0)
        coords = x_all[:, :3]
        coords_z = (coords - coords.mean(0, keepdim=True)) / (coords.std(0, keepdim=True) + 1e-6)

        feats = [coords_z]
        if self.base.use_patch:
            patches = x_all[:, 3:]
            feats.append(self.base.patch_enc(patches))
        emb_t = self.base.time_tok(t).to(x_all.device).view(1,-1).repeat(N,1)
        feats.append(emb_t)

        node_tokens = torch.cat(feats, dim=1)
        '''
        node_h = self.base.node_proj(node_tokens)
        # NOTE: no masking here; if base is graph-aware, override this class with a graph-aware base
        node_h = self.base.encoder(node_h.unsqueeze(0)).squeeze(0)
        i, j = edge_index
        hi, hj = node_h[i], node_h[j]
        coords_delta = coords_z[i] - coords_z[j]
        # base edge token WITHOUT p_in (will add later)
        edge_tok = torch.cat([
            dist_z.view(-1,1), sim_01.view(-1,1), coords_delta, hi, hj
        ], dim=1)
        '''
        node_h, coords_z = self.base.encode_nodes(x_all, t)
        i, j = edge_index
        edge_tok = torch.cat([dist_z.view(-1,1), sim_01.view(-1,1),
                              (coords_z[i]-coords_z[j]), node_h[i], node_h[j]], dim=1)
        
        return edge_tok, p_in

    def forward(self, x_all, edge_index, p_in, t, dist_z, sim_01):
        edge_tok_raw, p_in_vec = self._build_edge_tokens(x_all, edge_index, p_in, t, dist_z, sim_01)
        E = edge_tok_raw.size(0)
        H = self.edge_proj(edge_tok_raw).unsqueeze(0)  # (1,E,d)

        # Mask: only attend among edges that share nodes
        mask = edge_share_node_mask(edge_index)  # (E,E)
        for attn, ffn in zip(self.edge_attn, self.edge_ffn):
            H2, _ = attn(H, H, H, attn_mask=mask)
            H = H + H2
            H = H + ffn(H)

        H = H.squeeze(0)  # (E,d_edge)
        logits = self.edge_head(torch.cat([H, p_in_vec.view(-1,1)], dim=1)).view(-1)
        if self.predict_residual:
            # interpret sigmoid output in [-0.5,0.5] after centering, as residual
            delta = logits - 0.5
            return (p_in + delta).clamp(0,1)
        return logits


class TemporalPriorFusionWrapper(nn.Module):
    """
    Wrap a node-encoder (or edge-competition module) and add cross-attention from
    edge tokens to a 2-token memory {w, p_in} per graph/time.
    """
    def __init__(self, inner_edge_scorer: nn.Module, d_mem: int = 128, heads: int = 2,
                 predict_residual: bool = False):
        super().__init__()
        self.inner = inner_edge_scorer
        self.predict_residual = predict_residual
        self.mem_qproj = nn.Linear(1, d_mem)  # project scalar probs to memory dim
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_mem, num_heads=heads, batch_first=True)
        self.post = nn.Sequential(nn.Linear(d_mem, d_mem), nn.ReLU())

        # final head re-scores with memory context
        self.final = nn.Sequential(
            nn.Linear(d_mem + 1, d_mem), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_mem, 1), nn.Sigmoid()
        )

    def forward(self, x_all, edge_index, p_in, t, dist_z, sim_01):
        # We need w to build memory; recompute here to avoid threading changes
        w, _, _ = compute_prior_and_edgefeats({"x_all": x_all, "edge_index": edge_index})

        # First pass: get a provisional score from inner (used as edge token proxy)
        inner_out = self.inner(x_all, edge_index, p_in, t, dist_z, sim_01)  # (E,)

        # Build memory [w, p_in] -> (1, 2, d_mem)
        
        mem = torch.stack([w, p_in], dim=-1).unsqueeze(-1)  # (E, 2, 1)
        mem = self.mem_qproj(mem)                           # (E, 2, d_mem)
        mem = mem.mean(dim=0, keepdim=True)                 # (1, 2, d_mem)

        Q = self.mem_qproj(inner_out.view(-1,1)).unsqueeze(0)  # (1, E, d_mem)
        K = mem                                                # (1, 2, d_mem)
        V = mem
        ctx, _ = self.cross_attn(Q, K, V)                      # (1, E, d_mem)
        ctx = self.post(ctx).squeeze(0)                        # (E, d_mem)

        out = self.final(torch.cat([ctx, p_in.view(-1,1)], dim=1)).view(-1)
        if self.predict_residual:
            delta = out - 0.5
            return (p_in + delta).clamp(0,1)
        return out

class EdgeMLPHead(nn.Module):
    """
    Tiny model: scores edges from handcrafted features only.
    Inputs per-edge: [p_in, dist_z, sim_01, Δxyz(3)]
    (Optional) You can concat a light node summary by mean-pooling patches, but
    below is the minimal version.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + 1 + 1 + 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, x_all, edge_index, p_in, t, dist_z, sim_01):
        i, j = edge_index
        coords = x_all[:, :3]
        coords_z = (coords - coords.mean(0, keepdim=True)) / (coords.std(0, keepdim=True) + 1e-6)
        delta = coords_z[i] - coords_z[j]
        feats = torch.cat([
            p_in.view(-1,1),
            dist_z.view(-1,1),
            sim_01.view(-1,1),
            delta
        ], dim=1)
        return self.mlp(feats).view(-1)

class EdgeTinyAttnHead(nn.Module):
    """
    Tiny, fast model:
      1) Build a small per-edge token from handcrafted features
         [p_in, dist_z, sim_01, Δxyz(3)]  -> project to d_edge
      2) One layer of self-attention between edges that share a node
      3) Small MLP head -> sigmoid
    """
    def __init__(self, d_edge: int = 64, heads: int = 2, predict_residual: bool = False):
        super().__init__()
        self.predict_residual = predict_residual

        # 1) edge token builder (minimal features -> token)
        in_f = 1 + 1 + 1 + 3   # p_in, dist_z, sim_01, delta_xyz(3)
        self.edge_proj = nn.Sequential(
            nn.Linear(in_f, d_edge), nn.ReLU(),
            nn.Linear(d_edge, d_edge)
        )

        # 2) one tiny self-attention block (line-graph style)
        self.attn = nn.MultiheadAttention(embed_dim=d_edge, num_heads=heads, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_edge, d_edge*2), nn.ReLU(), nn.Linear(d_edge*2, d_edge)
        )

        # 3) small scoring head; concat p_in again for a skip scalar
        self.head = nn.Sequential(
            nn.Linear(d_edge + 1, d_edge), nn.ReLU(),
            nn.Linear(d_edge, 1), nn.Sigmoid()
        )

    def forward(self, x_all, edge_index, p_in, t, dist_z, sim_01):
        # build minimal edge features
        i, j = edge_index
        coords = x_all[:, :3]
        coords_z = (coords - coords.mean(0, keepdim=True)) / (coords.std(0, keepdim=True) + 1e-6)
        delta = coords_z[i] - coords_z[j]

        feats = torch.cat([
            p_in.view(-1,1),
            dist_z.view(-1,1),
            sim_01.view(-1,1),
            delta
        ], dim=1)                            # (E, 6)

        H = self.edge_proj(feats).unsqueeze(0)  # (1, E, d_edge)

        # mask: edges only attend to others that share a node
        mask = edge_share_node_mask(edge_index)  # (E,E) with 0/-inf

        # attn + residual + tiny FFN
        H2, _ = self.attn(H, H, H, attn_mask=mask)
        H = H + H2
        H = H + self.ffn(H)

        H = H.squeeze(0)                        # (E, d_edge)
        out = self.head(torch.cat([H, p_in.view(-1,1)], dim=1)).view(-1)

        if self.predict_residual:
            delta_p = out - 0.5
            return (p_in + delta_p).clamp(0, 1)
        return out


# --------------------------- Losses -----------------------------

def weighted_bce(pred_sigmoid: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
    wts = torch.where(target > 0.5,
                      torch.as_tensor(pos_weight, device=pred_sigmoid.device),
                      torch.as_tensor(1.0, device=pred_sigmoid.device))
    return (bce * wts).mean()

def focal_loss_sigmoid(pred_sigmoid: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    eps = 1e-8
    p_t = pred_sigmoid*target + (1 - pred_sigmoid)*(1 - target)
    alpha_t = alpha*target + (1 - alpha)*(1 - target)
    loss = -alpha_t * ((1 - p_t) ** gamma) * torch.log(p_t.clamp(min=eps))
    return loss.mean()

def edge_loss(pred_sigmoid: torch.Tensor, target: torch.Tensor, cfg: Config, pos_weight: Optional[float]) -> torch.Tensor:
    if cfg.loss_type.lower() == "bce":
        if pos_weight is None: pos_weight = 1.0
        return weighted_bce(pred_sigmoid, target, pos_weight)
    elif cfg.loss_type.lower() == "focal":
        # Map pos_weight -> alpha heuristic (rare pos -> larger alpha)
        if pos_weight is None:
            alpha = cfg.focal_alpha_base
        else:
            pos_prior = 1.0 / (1.0 + float(pos_weight))
            alpha = max(cfg.focal_alpha_base, 1.0 - pos_prior)
        return focal_loss_sigmoid(pred_sigmoid, target, alpha=alpha, gamma=cfg.focal_gamma)
    else:
        raise ValueError("loss_type must be 'bce' or 'focal'")

# --------------------------- Metrics ----------------------------

def binarize(p: torch.Tensor, thresh: float) -> torch.Tensor:
    return (p >= thresh).long()

def accuracy_balacc(pred_sig: torch.Tensor, y: torch.Tensor, thresh: float) -> Tuple[float, float, float, float]:
    yhat = binarize(pred_sig, thresh)
    tp = int(((yhat==1) & (y==1)).sum().item())
    tn = int(((yhat==0) & (y==0)).sum().item())
    fp = int(((yhat==1) & (y==0)).sum().item())
    fn = int(((yhat==0) & (y==1)).sum().item())
    tot = max(int(y.numel()), 1)
    acc = (tp + tn) / tot
    # balanced accuracy
    tpr = tp / max(tp+fn, 1e-9)
    tnr = tn / max(tn+fp, 1e-9)
    ba = 0.5 * (tpr + tnr)
    # precision/recall
    prec = tp / max(tp+fp, 1e-9)
    rec  = tp / max(tp+fn, 1e-9)
    return acc, ba, float(prec), float(rec)

# ------------------------ Inference -----------------------------

@torch.no_grad()
def reverse_step(model: nn.Module, x_all: torch.Tensor, edge_index: torch.Tensor, p_in: torch.Tensor,
                 t: int, dist_z: torch.Tensor, sim_01: torch.Tensor, damping: float = 0.6) -> torch.Tensor:
    p_hat = model(x_all, edge_index, p_in, t, dist_z, sim_01)
    p_next = (1.0 - damping) * p_in + damping * p_hat
    return p_next.clamp(0,1)

'''
def build_init_from_max_noise(sample: Dict[str,Any], E: int) -> Optional[torch.Tensor]:
    """Return soft init probs from max-noise carriers if available, else None."""
    # Preference order: explicit list of edges in max-noise, else 'R' mask
    if "max_noise_edge_index" in sample:
        idx_mask = torch.zeros(E, dtype=torch.bool, device=device)
        ei = sample["edge_index"]
        mnei = sample["max_noise_edge_index"]
        def to_pairs(t2): return (t2[0].long().tolist(), t2[1].long().tolist())
        s_i, s_j = to_pairs(ei); m_i, m_j = to_pairs(mnei)
        pairs_m = set(zip(m_i, m_j))
        for k,(a,b) in enumerate(zip(s_i, s_j)):
            if (a,b) in pairs_m:
                idx_mask[k] = True
        p_init = torch.zeros(E, device=device)
        p_init[idx_mask] = 0.9
        p_init[~idx_mask] = 0.1
        return p_init.clamp(0,1)

    if "R" in sample:
        R = sample["R"].view(-1)
        Rz = minmax01(-R)  # heuristic: smaller R -> larger init prob
        return Rz.clamp(0,1)

    return None
'''

def build_init_from_max_noise(sample: Dict[str,Any], E: int) -> Optional[torch.Tensor]:
    if "max_noise_edge_index" in sample:
        ei = sample["edge_index"].long()
        mnei = sample["max_noise_edge_index"].long()

        def undirected_pairs(t2):
            a, b = t2[0].tolist(), t2[1].tolist()
            return { (min(x,y), max(x,y)) for x,y in zip(a,b) }

        S = undirected_pairs(ei)
        M = undirected_pairs(mnei)

        idx_mask = torch.zeros(E, dtype=torch.bool, device=ei.device)
        # build a map from pair->index for speed
        pair2idx = {}
        for k,(x,y) in enumerate(zip(ei[0].tolist(), ei[1].tolist())):
            pair2idx[(min(x,y),max(x,y))] = k
        for pair in (S & M):
            idx_mask[pair2idx[pair]] = True

        p_init = torch.zeros(E, device=ei.device)
        p_init[idx_mask] = 0.9
        p_init[~idx_mask] = 0.1
        return p_init.clamp(0,1)

    if "R" in sample:
        R = sample["R"].view(-1)
        return minmax01(-R).clamp(0,1)

    return None


@torch.no_grad()
def run_inference_graph(model: nn.Module, sample: Dict[str,Any], cfg: Config, steps: int, damping: float) -> torch.Tensor:
    w, d_z, s_01 = compute_prior_and_edgefeats(sample)
    E = sample["edge_index"].size(1)

    if cfg.inference_mode == "max_noise":
        p0_init = build_init_from_max_noise(sample, E)
        p_cur = p0_init if p0_init is not None else (w + 0.01*torch.randn_like(w)).clamp(0,1)
    else:
        p_cur = (w + 0.01*torch.randn_like(w)).clamp(0,1)

    schedule = make_inference_schedule(T=cfg.T, steps=steps, mode="cosine")
    for t in schedule:
        p_cur = reverse_step(model, sample["x_all"], sample["edge_index"], p_cur, t, d_z, s_01, damping=damping)
    return p_cur
    
@torch.no_grad()
def eval_from_maxnoise(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    steps: Optional[int] = None,
    damping: Optional[float] = None,
    subset_frac: float = 1.0,
    max_batches: Optional[int] = None,
    seed: int = 1234,
) -> Tuple[float, float, float, int]:
    """
    Roll out from max-noise for each graph and return (mean_loss, mean_acc, mean_ba, n_graphs).
    Per-graph metrics are computed at cfg.thresh and then averaged equally across graphs.
    """
    steps = steps if steps is not None else (cfg.val_infer_steps or cfg.T)
    damping = damping if damping is not None else cfg.val_damping

    rng = random.Random(seed)
    loss_list, acc_list, ba_list = [], [], []
    used = 0

    for batch in loader:
        # optional subsampling
        if subset_frac < 1.0 and rng.random() > subset_frac:
            continue

        sample = batch[0]
        if sample.get("p0", None) is None:
            continue
        p0 = sample["p0"]; ei = sample["edge_index"]; x_all = sample["x_all"]
        if p0.numel() != ei.size(1):
            continue

        # rollout from max-noise
        p_pred = run_inference_graph(model, sample, cfg, steps=steps, damping=damping)

        # metrics per-graph
        l = edge_loss(p_pred, p0, cfg, pos_weight=None).item()
        acc, ba, _, _ = accuracy_balacc(p_pred, (p0 > 0.5).long(), cfg.thresh)

        loss_list.append(l); acc_list.append(acc); ba_list.append(ba)
        used += 1
        if max_batches is not None and used >= max_batches:
            break

    if used == 0:
        return float("nan"), float("nan"), float("nan"), 0

    return float(np.mean(loss_list)), float(np.mean(acc_list)), float(np.mean(ba_list)), used
    
    
def _is_aug1_file(path: str) -> bool:
    # matches "aug1" but not "aug10", "aug11", etc.
    name = os.path.basename(path).lower()
    return re.search(r"aug1(?!\d)", name) is not None

@torch.no_grad()
def save_val_aug1_prob_mats(model: nn.Module, cfg: Config, epoch: int) -> int:
    """
    For every validation .pt whose filename includes 'aug1' (but not aug10/aug11...),
    run rollout inference (no thresholding), then save:
      - A_prob: NxN soft adjacency (edge probabilities)
      - r:      Nx3 vertex coords (true_positions if present else x_all[:,:3])
    plus extra debug fields:
      - p_edge: (E,) predicted edge probabilities aligned to edge_index
      - edge_index: (2,E)
      - pt_path: saved as a string
    Output folder:
      <cfg.out_dir>/validation_on_full_image_epochXXX/
    """
    out_folder = os.path.join(cfg.out_dir, f"validation_on_full_image_epoch{epoch:03d}")
    ensure_dir(out_folder)

    # grab every validation .pt across run dirs
    val_paths = list_pt_files_across_runs(cfg.data_root, cfg.val_subdir, cfg.run_prefix)
    val_paths = [p for p in val_paths if _is_aug1_file(p)]

    if len(val_paths) == 0:
        print(f"[val_aug1] No aug1 files found in validation set.")
        return 0

    if sio is None:
        raise RuntimeError("scipy is required to save .mat files (scipy.io.savemat).")

    model.eval()
    saved = 0

    for pt_path in val_paths:
        data = load_pt_file(pt_path)

        # build sample dict compatible with run_inference_graph(...)
        sample = {
            "x_all": data.x.to(device),
            "edge_index": data.edge_index.to(device),
            "p0": (data.edge_label.to(device) if hasattr(data, "edge_label") and data.edge_label is not None else None),
            "path": pt_path,
        }
        if hasattr(data, "max_noise_edge_index"):
            sample["max_noise_edge_index"] = data.max_noise_edge_index.to(device)
        if hasattr(data, "R"):
            sample["R"] = data.R.to(device)
        if hasattr(data, "true_positions"):
            sample["true_positions"] = data.true_positions.to(device)

        # rollout inference (no thresholding)
        p_pred = run_inference_graph(
            model, sample, cfg,
            steps=(cfg.val_infer_steps or cfg.T),
            damping=cfg.val_damping
        )  # (E,)

        N = sample["x_all"].size(0)
        A_prob = build_soft_adjacency(p_pred, sample["edge_index"], N=N)  # (N,N)

        # coords: prefer true_positions if consistent
        coords = sample["x_all"][:, :3]
        if ("true_positions" in sample) and (sample["true_positions"].size(0) == N):
            coords = sample["true_positions"]

        base = os.path.splitext(os.path.basename(pt_path))[0]
        mat_path = os.path.join(out_folder, base + ".mat")

        sio.savemat(
            mat_path,
            {
                "A_prob": A_prob.detach().cpu().numpy(),
                "r": coords.detach().cpu().numpy(),
                "p_edge": p_pred.detach().cpu().numpy(),
                "edge_index": sample["edge_index"].detach().cpu().numpy(),
                "pt_path": pt_path,
            },
            do_compression=True
        )
        saved += 1

    print(f"[val_aug1] Saved {saved} .mat files to: {out_folder}")
    return saved


# ---------------------- Plotting Helpers ------------------------

def plot_metrics_dashboard_old(history: Dict[str, List[float]], out_png: str, title: str = "Training Dashboard"):
    keys = [k for k in history.keys() if isinstance(history[k], list) and len(history[k]) > 0]
    n = len(keys)
    if n == 0: return
    cols = 2
    rows = int(math.ceil(n/cols))
    plt.figure(figsize=(6*cols, 4*rows))
    for i,k in enumerate(keys, 1):
        plt.subplot(rows, cols, i)
        plt.plot(history[k], marker='o')
        plt.title(k); plt.xlabel("epoch"); plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_png, dpi=150)
    plt.close()
    
# ---- Matplotlib style helper (call once, e.g. inside train()) ----
def set_plot_style_ar12():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

def plot_metrics_dashboard(history: Dict[str, List[float]], out_png: str, title: str = "Training Dashboard"):
    """
    3x3 fixed layout:
      Row 1: train_loss, train_acc, train_ba
      Row 2: train_from_max_loss, train_from_max_acc, train_from_max_ba
      Row 3: val_loss, val_acc, val_ba

    Styling:
      - Arial 12 (via rcParams)
      - no grid
      - square subplot boxes
    """
    layout = [
        ["train_loss", "train_acc", "train_ba"],
        ["train_from_max_loss", "train_from_max_acc", "train_from_max_ba"],
        ["val_loss", "val_acc", "val_ba"],
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for r in range(3):
        for c in range(3):
            k = layout[r][c]
            ax = axes[r, c]

            ys = history.get(k, [])
            if isinstance(ys, list) and len(ys) > 0 and all(np.isfinite(np.array(ys, dtype=float))):
                ax.plot(ys, marker="o")
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")

            ax.set_title(k)
            ax.set_xlabel("epoch")
            ax.grid(False)

            # "axis square" for a line plot = square box (not equal data scaling)
            try:
                ax.set_box_aspect(1)  # matplotlib >= 3.3
            except Exception:
                ax.set_aspect("auto")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    
@torch.no_grad()
def _pick_nonempty_val_sample(val_loader) -> Optional[Dict[str, Any]]:
    """Return first val sample that has labels and at least one positive edge."""
    for batch in val_loader:
        sample = batch[0]
        p0 = sample.get("p0", None)
        ei = sample["edge_index"]
        if p0 is None: 
            continue
        if p0.numel() != ei.size(1): 
            continue
        if (p0 > 0.5).sum().item() > 0:
            return sample
    return None

def _rasterize_edge_mip(coords_xyz: torch.Tensor,
                        edge_index: torch.Tensor,
                        edge_mask: torch.Tensor,
                        axis: str = "z",
                        max_edges: int = 40000) -> np.ndarray:
    """
    Build a 2D max-projection of edges by line-drawing between node endpoints.
    coords_xyz: (N,3) float tensor in voxel/pixel units (will be rounded)
    edge_mask: (E,) bool tensor selecting which edges to draw
    Returns a uint16 2D image (counts per pixel).
    """
    # select edges
    ei = edge_index[:, edge_mask]
    i, j = ei[0].cpu().numpy(), ei[1].cpu().numpy()

    # shift coords to positive and round
    C = coords_xyz.detach().cpu().numpy()
    C = C - C.min(axis=0, keepdims=True)
    C = np.round(C).astype(np.int32)

    # infer size
    xmax, ymax, zmax = C[:,0].max(), C[:,1].max(), C[:,2].max()
    H, W = (ymax + 1, xmax + 1)   # for 'z' projection we keep (y,x) plane

    # safety cap for huge graphs
    E = i.shape[0]
    if E > max_edges:
        sel = np.random.choice(E, size=max_edges, replace=False)
        i, j = i[sel], j[sel]

    # set up target buffer
    img = np.zeros((H, W), dtype=np.uint16)

    # choose which plane to draw on
    ax = axis.lower()
    if ax not in ("x","y","z"): ax = "z"

    # simple integer sampling along the 3D segment; drop the projection axis
    def draw_edge(a, b):
        pa = C[a]; pb = C[b]
        steps = int(max(abs(pb - pa)) + 1)
        if steps <= 1:
            yy, xx = pa[1], pa[0]
            if 0 <= yy < H and 0 <= xx < W: img[yy, xx] = min(img[yy, xx] + 1, np.iinfo(img.dtype).max)
            return
        ts = np.linspace(0.0, 1.0, steps)
        P = np.round(pa[None,:] + (pb - pa)[None,:] * ts[:,None]).astype(np.int32)
        if ax == "z":
            yy, xx = P[:,1], P[:,0]
        elif ax == "y":
            # project onto (z,x)
            yy, xx = P[:,2], P[:,0]
        else:  # ax == "x"
            # project onto (z,y)
            yy, xx = P[:,2], P[:,1]
        # clip-safe writes
        m = (yy >= 0)&(yy < img.shape[0])&(xx >= 0)&(xx < img.shape[1])
        yy, xx = yy[m], xx[m]
        # increment counts
        img[yy, xx] = np.minimum(img[yy, xx] + 1, np.iinfo(img.dtype).max)

    for a, b in zip(i, j):
        draw_edge(a, b)

    return img

@torch.no_grad()
def save_val_rollout_panel(model: nn.Module,
                           val_loader: DataLoader,
                           cfg: Config,
                           out_png: str,
                           axis: str = "z"):
    """
    Pick a non-empty val graph, run rollout, make 3-panel visualization:
    [Pred MIP | GT MIP | Overlay]. Also prints per-graph ACC/BA in title.
    """
    sample = _pick_nonempty_val_sample(val_loader)
    if sample is None:
        # no valid sample found; create a placeholder
        plt.figure(figsize=(10,3))
        plt.text(0.5, 0.5, "No non-empty validation graph found.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()
        return

    # rollout prediction
    p_pred = run_inference_graph(
        model, sample, cfg,
        steps=(cfg.val_infer_steps or cfg.T),
        damping=cfg.val_damping
    )

    # metrics on this specific subgraph
    y = (sample["p0"] > 0.5).long()
    acc, ba, _, _ = accuracy_balacc(p_pred, y, cfg.thresh)

    # threshold masks
    pred_mask = (p_pred >= cfg.thresh)
    gt_mask   = (y == 1)

    coords = sample["x_all"][:, :3]

    # build 2D MIPs
    mip_pred = _rasterize_edge_mip(coords, sample["edge_index"], pred_mask, axis=axis)
    mip_gt   = _rasterize_edge_mip(coords, sample["edge_index"], gt_mask,   axis=axis)

    # normalize to [0,1] for display
    def norm01(a):
        a = a.astype(np.float32)
        return a / (a.max() + 1e-6)

    A = norm01(mip_pred)
    G = norm01(mip_gt)

    # overlay color coding:
    # TP = min(A,G) in white channel; FP = A-G>0 in magenta; FN = G-A>0 in green
    TP = np.minimum(A, G)
    FP = np.clip(A - G, 0, 1)
    FN = np.clip(G - A, 0, 1)

    overlay = np.stack([
        # R: FP + TP
        np.clip(FP + TP, 0, 1),
        # G: FN + TP
        np.clip(FN + TP, 0, 1),
        # B: TP only (kept for white on TP)
        np.clip(TP, 0, 1)
    ], axis=-1)

    # plot panel
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.imshow(A, origin="lower"); plt.title("Predicted MIP"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(G, origin="lower"); plt.title("Ground Truth MIP"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(overlay, origin="lower"); plt.title("Overlay (TP white, FP magenta, FN green)"); plt.axis("off")
    plt.suptitle(f"Val rollout @ thresh={cfg.thresh:.2f}  |  ACC={acc:.3f}  BA={ba:.3f}\n{os.path.basename(sample.get('path',''))}")
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig(out_png, dpi=150)
    plt.close()
    
# ============================
# Plot style + dashboards
# ============================

def set_plot_style_ar12():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

def _safe_box_aspect(ax):
    try:
        ax.set_box_aspect(1)  # axis square
    except Exception:
        pass

def plot_dashboard_combined_3(history: Dict[str, List[float]], out_png: str, title: str):
    """
    3-panel dashboard:
      - Loss
      - Accuracy
      - Balanced Accuracy

    Train(max rollout) = blue, Val = orange (matplotlib defaults by call order)
    Lines only (no markers)
    Accuracy/BA y-limits fixed to [0, 1]
    Axis square (box aspect 1)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("loss", "train_loss", "val_loss", None),
        ("acc",  "train_acc",  "val_acc",  (0.0, 1.0)),
        ("ba",   "train_ba",   "val_ba",   (0.0, 1.0)),
    ]

    for ax, (name, k_tr, k_va, ylim) in zip(axes, panels):
        y_tr = np.asarray(history.get(k_tr, []), dtype=float)
        y_va = np.asarray(history.get(k_va, []), dtype=float)

        x_tr = np.arange(len(y_tr))
        x_va = np.arange(len(y_va))

        plotted_any = False
        if y_tr.size > 0 and np.isfinite(y_tr).any():
            ax.plot(x_tr, y_tr, linewidth=2.0, label="train (max rollout)")
            plotted_any = True
        if y_va.size > 0 and np.isfinite(y_va).any():
            ax.plot(x_va, y_va, linewidth=2.0, label="val")
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")

        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(False)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)
        _safe_box_aspect(ax)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def _slice_last_n(y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.arange(0), y
    if n is None or n <= 0 or y.size <= n:
        y2 = y
    else:
        y2 = y[-n:]
    x = np.arange(y2.size)
    return x, y2

def _auto_ylim_for_two_curves(y1: np.ndarray, y2: np.ndarray, pad_frac: float = 0.08, pad_min: float = 0.02):
    """
    Auto y-lims that clearly show both curves.
    Ignores non-finite entries.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    yy = np.concatenate([y1[np.isfinite(y1)], y2[np.isfinite(y2)]], axis=0) if (y1.size+y2.size) > 0 else np.array([])
    if yy.size == 0:
        return None
    lo = float(np.min(yy))
    hi = float(np.max(yy))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi - lo < 1e-9:
        # flat curves; expand a bit
        pad = max(pad_min, 0.05 * max(abs(hi), 1.0))
        return (lo - pad, hi + pad)
    pad = max(pad_min, pad_frac * (hi - lo))
    return (lo - pad, hi + pad)

def plot_dashboard_last10_3(history: Dict[str, List[float]], out_png: str, title: str, n_last: int = 10):
    """
    Last-N (default 10) dashboard:
      - Loss
      - Accuracy
      - Balanced Accuracy

    Train(max rollout) vs Val.
    No smoothing.
    For acc/BA: AUTO y-lims (NOT fixed 0..1) so both curves are clearly visible.
    Axis square (box aspect 1)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("loss", "train_loss", "val_loss", None),   # auto y-lim by default
        ("acc",  "train_acc",  "val_acc",  "auto"),
        ("ba",   "train_ba",   "val_ba",   "auto"),
    ]

    for ax, (name, k_tr, k_va, ylim_mode) in zip(axes, panels):
        y_tr_full = np.asarray(history.get(k_tr, []), dtype=float)
        y_va_full = np.asarray(history.get(k_va, []), dtype=float)

        x_tr, y_tr = _slice_last_n(y_tr_full, n_last)
        x_va, y_va = _slice_last_n(y_va_full, n_last)

        plotted_any = False
        if y_tr.size > 0 and np.isfinite(y_tr).any():
            ax.plot(x_tr, y_tr, linewidth=2.0, label="train (max rollout)")
            plotted_any = True
        if y_va.size > 0 and np.isfinite(y_va).any():
            ax.plot(x_va, y_va, linewidth=2.0, label="val")
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")

        shown_n = min(n_last, max(len(y_tr_full), len(y_va_full)))
        ax.set_title(f"{name} (last {shown_n})")
        ax.set_xlabel("recent epoch index")
        ax.grid(False)
        ax.legend(frameon=False)

        if ylim_mode == "auto":
            yl = _auto_ylim_for_two_curves(y_tr, y_va)
            if yl is not None:
                ax.set_ylim(*yl)

        _safe_box_aspect(ax)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================
# Max-rollout eval (train subset + val)
# ============================

@torch.no_grad()
def infer_edges_from_max_rollout(model: nn.Module,
                                 sample: Dict[str, Any],
                                 cfg: Config,
                                 damping: float,
                                 infer_steps: Optional[int] = None) -> torch.Tensor:
    """
    Run reverse diffusion starting from max-noise init (if available),
    otherwise from prior w (or p0 if you switch inference_mode elsewhere).

    Returns:
      p_pred in [0,1] for the sample's edge_index.
    """
    x_all = sample["x_all"]
    edge_index = sample["edge_index"]
    E = edge_index.size(1)

    w, dist_z, sim_01 = compute_prior_and_edgefeats({"x_all": x_all, "edge_index": edge_index})

    # init p_in (max-noise carriers preferred)
    p_init = None
    if cfg.inference_mode == "max_noise":
        p_init = build_init_from_max_noise(sample, E)
    if p_init is None:
        # fallback: prior
        p_init = w

    p = p_init.clamp(0, 1)

    # choose how many reverse steps we take (number of t-values to visit)
    steps = int(infer_steps) if infer_steps is not None else (int(cfg.val_infer_steps) if cfg.val_infer_steps is not None else int(cfg.T))
    ts = make_inference_schedule(T=int(cfg.T), steps=max(1, steps), mode="cosine")

    for t in ts:
        p = reverse_step(model, x_all, edge_index, p, t=int(t), dist_z=dist_z, sim_01=sim_01, damping=float(damping))

    return p.clamp(0, 1)

@torch.no_grad()
def eval_max_rollout_paths(model: nn.Module,
                           paths: List[str],
                           cfg: Config,
                           device: torch.device,
                           max_batches: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Evaluate loss/acc/BA by running inference-from-max-rollout on each graph.
    Returns mean over evaluated graphs.
    """
    model.eval()

    losses, accs, bas = [], [], []

    ds = GraphPathsDataset(paths, device=device)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_list)

    nb = 0
    for batch in dl:
        if max_batches is not None and nb >= int(max_batches):
            break
        nb += 1

        sample = batch[0]
        if sample.get("p0", None) is None:
            continue  # can't score without labels

        y = sample["p0"].view(-1)
        # binarize labels if needed
        y_bin = (y > 0.5).long()

        # class weights for loss
        pos = float((y_bin == 1).sum().item())
        neg = float((y_bin == 0).sum().item())
        if pos < 1:
            continue
        pos_weight = (neg / max(pos, 1.0)) if cfg.loss_type.lower() == "bce" else (neg / max(pos, 1.0))

        p_pred = infer_edges_from_max_rollout(
            model=model,
            sample=sample,
            cfg=cfg,
            damping=float(cfg.val_damping),
            infer_steps=cfg.val_infer_steps
        )

        # base edge loss
        loss = edge_loss(p_pred, y_bin.float(), cfg, pos_weight=pos_weight)

        # optional penalties (match your training settings)
        if cfg.loop_lambda is not None and float(cfg.loop_lambda) > 0 and cfg.loop_penalty != "none":
            A = build_soft_adjacency(p_pred, sample["edge_index"], N=sample["x_all"].size(0))
            if cfg.loop_penalty == "simple":
                loss = loss + float(cfg.loop_lambda) * triangle_penalty(A)
            elif cfg.loop_penalty == "complex":
                loss = loss + float(cfg.loop_lambda) * resolvent_cycle_penalty(
                    A, tau=float(cfg.loop_tau),
                    estimator=str(cfg.loop_complex_estimator),
                    R=int(cfg.loop_hutch_R),
                )
            elif cfg.loop_penalty == "nbt":
                loss = loss + float(cfg.loop_lambda) * nonbacktracking_loop_penalty(
                    edge_index=sample["edge_index"].long(),
                    p_edge=p_pred,
                    N_nodes=int(sample["x_all"].size(0)),
                    K=int(cfg.loop_K),
                    R=int(cfg.loop_hutch_R),
                    weighting=str(cfg.loop_nbt_weighting),
                )

        if cfg.endpoint_lambda is not None and float(cfg.endpoint_lambda) > 0 and cfg.endpoint_penalty != "none":
            A = build_soft_adjacency(p_pred, sample["edge_index"], N=sample["x_all"].size(0))
            node_mask = None
            if bool(cfg.endpoint_mask_gt):
                node_mask = gt_node_mask_from_edges(sample["edge_index"], y_bin, int(sample["x_all"].size(0)))
            if cfg.endpoint_penalty == "bump":
                ep = endpoint_bump_penalty(A, sigma=float(cfg.endpoint_sigma), node_mask=node_mask)
                loss = loss + float(cfg.endpoint_lambda) * ep
            elif cfg.endpoint_penalty == "deg2":
                ep = degree_to_two_penalty(A, node_mask=node_mask)
                loss = loss + float(cfg.endpoint_lambda) * ep
            elif cfg.endpoint_penalty == "quad":
                ep = quadratic_edge_penalty(A)
                loss = loss + float(cfg.endpoint_lambda) * ep

        acc, ba, _, _ = accuracy_balacc(p_pred, y_bin, cfg.thresh)

        losses.append(float(loss.item()))
        accs.append(float(acc))
        bas.append(float(ba))

    if len(losses) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(accs)), float(np.mean(bas))


# ============================
# Epoch-0 baseline (exactly like your GNN script)
# ============================

@torch.no_grad()
def maybe_add_epoch0_baseline(cfg: Config,
                             model_factory,
                             history: Dict[str, List[float]],
                             train_paths: List[str],
                             val_paths: List[str],
                             device: torch.device):
    """
    Adds an epoch-0 baseline (init model) to history ONCE per run folder,
    even if resuming. Uses:
      - init weights saved in cfg.out_dir/init_model_state.pt (created if missing)
      - a flag file cfg.out_dir/baseline_epoch0_done.txt to avoid duplicates

    Baseline values are PREPENDED (insert at index 0):
      0 = init baseline, 1.. = trained epochs
    """
    DO_PRETRAIN_BASELINE = True
    if not DO_PRETRAIN_BASELINE:
        return

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "figs"))

    flag_path = os.path.join(cfg.out_dir, "baseline_epoch0_done.txt")
    init_path = os.path.join(cfg.out_dir, "init_model_state.pt")

    if os.path.exists(flag_path):
        return

    # Build a temporary model for baseline eval (does NOT affect your training model)
    model0 = model_factory().to(device)

    # Load or create deterministic init weights
    if os.path.exists(init_path):
        model0.load_state_dict(torch.load(init_path, map_location=device))
    else:
        devices = [torch.device("cuda")] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            if cfg.seed is not None:
                seed_everything(int(cfg.seed))
            else:
                seed_everything(42)
            model0 = model_factory().to(device)
        torch.save(model0.state_dict(), init_path)

    print("[baseline] Computing epoch-0 (init) metrics and prepending to history...")

    # Train(max rollout) baseline: subset + cap (same knobs you already have)
    # Use a prefix subset based on train_eval_frac to keep it quick/deterministic
    if len(train_paths) > 0:
        n_take = max(1, int(round(float(cfg.train_eval_frac) * len(train_paths))))
        train_subset = train_paths[:n_take]
    else:
        train_subset = []

    tr0_loss, tr0_acc, tr0_ba = eval_max_rollout_paths(
        model0, train_subset, cfg, device,
        max_batches=cfg.train_eval_max_batches
    )
    va0_loss, va0_acc, va0_ba = eval_max_rollout_paths(
        model0, val_paths, cfg, device,
        max_batches=None
    )

    # Ensure keys exist
    for k in ("train_loss","train_acc","train_ba","val_loss","val_acc","val_ba"):
        history.setdefault(k, [])

    # Prepend epoch-0
    history["train_loss"].insert(0, tr0_loss)
    history["train_acc"].insert(0, tr0_acc)
    history["train_ba"].insert(0, tr0_ba)
    history["val_loss"].insert(0, va0_loss)
    history["val_acc"].insert(0, va0_acc)
    history["val_ba"].insert(0, va0_ba)

    # Save epoch-0 dashboards
    dash0_png = os.path.join(cfg.out_dir, "figs", "dashboard_combined_epoch_000.png")
    plot_dashboard_combined_3(history, dash0_png, title="Graph Diffusion Combined Dashboard @ epoch 0")

    dash0_last10 = os.path.join(cfg.out_dir, "figs", "dashboard_last10_epoch_000.png")
    plot_dashboard_last10_3(history, dash0_last10, title="Graph Diffusion Last-10 Dashboard @ epoch 0", n_last=10)

    with open(flag_path, "w") as f:
        f.write("baseline epoch0 added\n")



# ---------------------- Trace-style 3D plotting ------------------------

def _set_axes_equal_3d(ax):
    """Make 3D plot axes have equal scale (like MATLAB axis equal)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    ax.set_xlim3d(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim3d(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim3d(z_mid - max_range/2, z_mid + max_range/2)


def _build_binary_adj_from_mask(edge_index: torch.Tensor,
                                edge_mask: torch.Tensor,
                                N: int,
                                device: torch.device) -> torch.Tensor:
    """
    Build a symmetric 0/1 adjacency matrix A (N x N) for the edges
    selected by edge_mask.
    """
    A = torch.zeros((N, N), dtype=torch.float32, device=device)
    if edge_mask.sum().item() == 0:
        return A

    ei = edge_index[:, edge_mask]
    i, j = ei[0], ei[1]
    A.index_put_((i, j), torch.ones_like(i, dtype=torch.float32), accumulate=False)
    A.index_put_((j, i), torch.ones_like(j, dtype=torch.float32), accumulate=False)
    A.fill_diagonal_(0.0)
    return A


def plot_am_py(AM: np.ndarray,
               r: np.ndarray,
               color: str = "tab:blue",
               ax: Optional[Any] = None,
               linewidth: float = 1.0):
    """
    Python analogue of MATLAB PlotAM(AM, r), but with a single, deterministic color.

    AM : (N,N) adjacency matrix (values >0 treated as edges)
    r  : (N,3) vertex coordinates
    color : matplotlib color spec for all edges
    ax : optional existing 3D axis; if None, a new one is created
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # symmetrize and keep upper triangle (like MATLAB code)
    AM = np.maximum(AM, AM.T)
    AMu = np.triu(AM)

    # find all nonzero entries (edges)
    i_idx, j_idx = np.nonzero(AMu)
    if len(i_idx) == 0:
        return ax

    # draw each edge as a 3D line
    for i, j in zip(i_idx, j_idx):
        X = [r[i, 0], r[j, 0]]
        Y = [r[i, 1], r[j, 1]]
        Z = [r[i, 2], r[j, 2]]
        ax.plot(Y, X, Z, color=color, linewidth=linewidth)  # note (Y,X,Z) to mimic MATLAB orientation

    return ax

'''
@torch.no_grad()
def save_val_rollout_trace_panel(model: nn.Module,
                                 val_loader: DataLoader,
                                 cfg: Config,
                                 out_png: str,
                                 color_pred: str = "tab:blue",
                                 color_gt: str = "tab:orange"):
    sample = _pick_nonempty_val_sample(val_loader)
    if sample is None:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No non-empty validation graph found.",
                 ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()
        return

    # rollout prediction from max-noise
    p_pred = run_inference_graph(
        model, sample, cfg,
        steps=(cfg.val_infer_steps or cfg.T),
        damping=cfg.val_damping
    )

    y = (sample["p0"] > 0.5).long()
    pred_mask = (p_pred >= cfg.thresh)
    gt_mask   = (y == 1)

    # --- NEW: choose coords safely ---
    x_coords = sample["x_all"][:, :3]
    N_nodes  = x_coords.size(0)

    if "true_positions" in sample and sample["true_positions"].size(0) == N_nodes:
        coords = sample["true_positions"]
    else:
        # fall back if true_positions is missing or mismatched
        coords = x_coords

    device_local = coords.device

    # --- build adjacency with N_nodes derived from x_all / edge_index ---
    A_gt   = _build_binary_adj_from_mask(sample["edge_index"], gt_mask,   N_nodes, device_local)
    A_pred = _build_binary_adj_from_mask(sample["edge_index"], pred_mask, N_nodes, device_local)

    A_gt_np   = A_gt.cpu().numpy()
    A_pred_np = A_pred.cpu().numpy()
    r_np      = coords.cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_am_py(A_gt_np,   r_np, color=color_gt,   ax=ax, linewidth=1.0)
    plot_am_py(A_pred_np, r_np, color=color_pred, ax=ax, linewidth=1.5)

    _set_axes_equal_3d(ax)
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_zlabel("Z")
    title_path = os.path.basename(sample.get("path", ""))
    ax.set_title(f"Val trace overlay @ thresh={cfg.thresh:.2f}\n{title_path}")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
'''

'''
@torch.no_grad()
def save_val_rollout_trace_panel(model: nn.Module,
                                 val_loader: DataLoader,
                                 cfg: Config,
                                 out_png: str,
                                 color_pred: str = "tab:blue",
                                 color_gt: str = "tab:orange"):
    """
    Pick a non-empty val graph, run rollout, and plot 3D traces:
    - Left subplot: 3D view (same as before).
    - Right subplot: top-down XY view (camera along +Z).
    """
    sample = _pick_nonempty_val_sample(val_loader)
    if sample is None:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No non-empty validation graph found.",
                 ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()
        return

    # rollout prediction from max-noise (same as qual panel)
    p_pred = run_inference_graph(
        model, sample, cfg,
        steps=(cfg.val_infer_steps or cfg.T),
        damping=cfg.val_damping
    )

    # Thresholded masks
    y = (sample["p0"] > 0.5).long()
    pred_mask = (p_pred >= cfg.thresh)
    gt_mask   = (y == 1)

    # Choose coordinates safely: prefer true_positions if present and consistent
    x_coords = sample["x_all"][:, :3]
    N_nodes  = x_coords.size(0)

    if ("true_positions" in sample) and (sample["true_positions"].size(0) == N_nodes):
        coords = sample["true_positions"]
    else:
        coords = x_coords

    device_local = coords.device

    # Build adjacency matrices for GT and prediction
    A_gt   = _build_binary_adj_from_mask(sample["edge_index"], gt_mask,   N_nodes, device_local)
    A_pred = _build_binary_adj_from_mask(sample["edge_index"], pred_mask, N_nodes, device_local)

    A_gt_np   = A_gt.cpu().numpy()
    A_pred_np = A_pred.cpu().numpy()
    r_np      = coords.cpu().numpy()

    title_path = os.path.basename(sample.get("path", ""))

    # --- Figure with two subplots ---
    fig = plt.figure(figsize=(12, 6))

    # Left: standard 3D view (same as original behavior)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_am_py(A_gt_np,   r_np, color=color_gt,   ax=ax1, linewidth=1.0)
    plot_am_py(A_pred_np, r_np, color=color_pred, ax=ax1, linewidth=1.5)
    _set_axes_equal_3d(ax1)
    ax1.set_xlabel("Y")
    ax1.set_ylabel("X")
    ax1.set_zlabel("Z")
    ax1.set_title("3D view")

    # Right: top-down XY view (Z axis toward camera)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_am_py(A_gt_np,   r_np, color=color_gt,   ax=ax2, linewidth=1.0)
    plot_am_py(A_pred_np, r_np, color=color_pred, ax=ax2, linewidth=1.5)
    _set_axes_equal_3d(ax2)
    # Look straight down the +Z axis: elevation 90 deg
    ax2.view_init(elev=90, azim=-90)  # azim can be tweaked; -90 gives X right, Y up
    ax2.set_xlabel("Y")
    ax2.set_ylabel("X")
    ax2.set_zlabel("")       # hide Z label for projection-like look
    ax2.set_zticks([])       # hide Z ticks
    ax2.set_title("XY projection view")

    # Global title
    fig.suptitle(
        f"Val trace overlay @ thresh={cfg.thresh:.2f}\n{title_path}",
        y=0.95
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_png, dpi=150)
    plt.close()
'''

@torch.no_grad()
def save_val_rollout_trace_panel(model: nn.Module,
                                 val_loader: DataLoader,
                                 cfg: Config,
                                 out_png: str,
                                 color_pred: str = "tab:blue",
                                 color_gt: str = "tab:blue"):
    """
    Pick a non-empty val graph, run rollout, and plot 3D traces in top-down view:
    - Left subplot: ground truth trace (XY projection, Z toward camera)
    - Right subplot: predicted trace (XY projection, Z toward camera)
    Grid lines removed on both.
    OLD: color_gt: str = "tab:orange"
    """
    sample = _pick_nonempty_val_sample(val_loader)
    if sample is None:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No non-empty validation graph found.",
                 ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()
        return

    # Rollout prediction from max-noise
    p_pred = run_inference_graph(
        model, sample, cfg,
        steps=(cfg.val_infer_steps or cfg.T),
        damping=cfg.val_damping
    )

    # Thresholded masks
    y = (sample["p0"] > 0.5).long()
    pred_mask = (p_pred >= cfg.thresh)
    gt_mask   = (y == 1)

    # Choose coordinates safely: prefer true_positions if present and consistent
    x_coords = sample["x_all"][:, :3]
    N_nodes  = x_coords.size(0)

    if ("true_positions" in sample) and (sample["true_positions"].size(0) == N_nodes):
        coords = sample["true_positions"]
    else:
        coords = x_coords

    device_local = coords.device

    # Build adjacency matrices for GT and prediction
    A_gt   = _build_binary_adj_from_mask(sample["edge_index"], gt_mask,   N_nodes, device_local)
    A_pred = _build_binary_adj_from_mask(sample["edge_index"], pred_mask, N_nodes, device_local)

    A_gt_np   = A_gt.cpu().numpy()
    A_pred_np = A_pred.cpu().numpy()
    r_np      = coords.cpu().numpy()

    title_path = os.path.basename(sample.get("path", ""))

    # --- Figure with two subplots: GT vs Pred, both top-down XY ---
    fig = plt.figure(figsize=(12, 6))

    # Left: ground truth only
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_am_py(A_gt_np, r_np, color=color_gt, ax=ax1, linewidth=1.5)
    _set_axes_equal_3d(ax1)
    ax1.view_init(elev=90, azim=-90)   # top-down: Z toward camera
    ax1.set_xlabel("Y")
    ax1.set_ylabel("X")
    ax1.set_zlabel("")
    ax1.set_zticks([])
    ax1.grid(False)
    ax1.set_title("Ground truth (XY top view)")

    # Right: prediction only
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_am_py(A_pred_np, r_np, color=color_pred, ax=ax2, linewidth=1.5)
    _set_axes_equal_3d(ax2)
    ax2.view_init(elev=90, azim=-90)   # same top-down view
    ax2.set_xlabel("Y")
    ax2.set_ylabel("X")
    ax2.set_zlabel("")
    ax2.set_zticks([])
    ax2.grid(False)
    ax2.set_title("Prediction (XY top view)")

    # Global title
    fig.suptitle(
        f"Val trace @ thresh={cfg.thresh:.2f}\n{title_path}",
        y=0.95
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_png, dpi=150)
    plt.close()



# ----------------------- Checkpointing --------------------------

def save_checkpoint(out_dir: str, cfg: Config, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, step: int, history: Dict[str, List[float]], fname: Optional[str] = None):
    ensure_dir(out_dir)
    if fname is None: fname = cfg.ckpt_name
    ckpt = {
        "cfg": asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "history": history
    }
    torch.save(ckpt, os.path.join(out_dir, fname))

def try_resume(cfg: Config, model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int,int,Dict[str,List[float]]]:
    path = os.path.join(cfg.out_dir, cfg.ckpt_name)
    if cfg.resume_if_found and os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch = int(ckpt.get("epoch", 0))
        step  = int(ckpt.get("step", 0))
        history = ckpt.get("history", {})
        print(f"[resume] Loaded checkpoint from {path} @ epoch={epoch}, step={step}")
        return epoch, step, history
    return 0, 0, {
    "train_loss": [], "train_acc": [], "train_ba": [],
    "val_loss":   [], "val_acc":   [], "val_ba":   [],
    "train_from_max_loss": [], "train_from_max_acc": [], "train_from_max_ba": []
    }

# ------------------------- Training Loop ------------------------

def train(cfg: Config):
    print("Device:", device)

    ensure_dir(cfg.out_dir)
    fig_dir = os.path.join(cfg.out_dir, "figs"); ensure_dir(fig_dir)
    set_plot_style_ar12()

    train_paths = list_pt_files_across_runs(cfg.data_root, cfg.train_subdir, cfg.run_prefix)
    val_paths   = list_pt_files_across_runs(cfg.data_root, cfg.val_subdir, cfg.run_prefix)
    train_paths = train_paths[:int(len(train_paths)*cfg.train_frac)]
    val_paths   = val_paths[:int(len(val_paths)*cfg.val_frac)]
    print(f"Train files: {len(train_paths)} | Val files: {len(val_paths)}")

    train_ds = GraphPathsDataset(train_paths, device)
    val_ds   = GraphPathsDataset(val_paths, device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_graphs, shuffle=True, collate_fn=collate_list)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_list)

    # Peek Fdim to set patch_dim
    tmp_sample = load_pt_file(train_paths[0]) if len(train_paths)>0 else load_pt_file(val_paths[0])
    Fdim = tmp_sample.x.size(1)
    patch_dim = max(Fdim - cfg.coord_dim, 0)

    # Model & optimizer
    def build_model(cfg: Config, patch_dim: int) -> nn.Module:
        if cfg.arch == "edge_mlp":
            return EdgeMLPHead(hidden=64).to(device)
        if cfg.arch == "edge_mlp_attn":
            # tiny defaults; tweak via cfg if you like
            return EdgeTinyAttnHead(
                d_edge=64, heads=max(1, min(2, cfg.edge_mha_heads)),
                predict_residual=cfg.predict_residual
            ).to(device)
            
        # 1) base node encoder
        if cfg.arch in ("baseline", "edgecomp", "temporal", "combo"):
            base = NodeTransformerEdgeHead(coord_dim=cfg.coord_dim, patch_dim=patch_dim,
                                           node_emb_dim=cfg.node_patch_emb_dim, d_model=cfg.d_model,
                                           nhead=cfg.nhead, nlayers=cfg.nlayers, time_dim=cfg.time_dim,
                                           edge_head_hidden=cfg.edge_head_hidden)
        elif cfg.arch == "graphaware":
            base = GraphAwareNodeTransformerEdgeHead(coord_dim=cfg.coord_dim, patch_dim=patch_dim,
                                                     node_emb_dim=cfg.node_patch_emb_dim, d_model=cfg.d_model,
                                                     nhead=cfg.nhead, nlayers=cfg.nlayers, time_dim=cfg.time_dim,
                                                     edge_head_hidden=cfg.edge_head_hidden,
                                                     graph_k=cfg.graph_k)
        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")

        # 2) compose wrappers
        model: nn.Module = base
        if cfg.arch in ("edgecomp", "combo"):
            model = EdgeCompetitionWrapper(model,
                                           d_edge=max(256, cfg.d_model*2),
                                           heads=cfg.edge_mha_heads,
                                           layers=cfg.edge_mha_layers,
                                           predict_residual=cfg.predict_residual)
        if cfg.arch in ("temporal", "combo"):
            model = TemporalPriorFusionWrapper(model,
                                               d_mem=max(128, cfg.d_model),
                                               heads=cfg.temporal_mha_heads,
                                               predict_residual=cfg.predict_residual)
        return model

    model = build_model(cfg, patch_dim).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    def model_factory():
        # IMPORTANT: build the exact same model architecture you train with
        if cfg.arch == "baseline":
            return NodeTransformerEdgeHead(
                coord_dim=cfg.coord_dim,
                patch_dim=patch_dim,
                node_emb_dim=cfg.node_patch_emb_dim,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                nlayers=cfg.nlayers,
                time_dim=cfg.time_dim,
                edge_head_hidden=cfg.edge_head_hidden
            )
        elif cfg.arch == "graphaware":
            return GraphAwareNodeTransformerEdgeHead(
                coord_dim=cfg.coord_dim,
                patch_dim=patch_dim,
                node_emb_dim=cfg.node_patch_emb_dim,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                nlayers=cfg.nlayers,
                time_dim=cfg.time_dim,
                edge_head_hidden=cfg.edge_head_hidden,
                graph_k=cfg.graph_k
            )
        elif cfg.arch == "edgecomp":
            base = NodeTransformerEdgeHead(
                coord_dim=cfg.coord_dim, patch_dim=patch_dim,
                node_emb_dim=cfg.node_patch_emb_dim, d_model=cfg.d_model,
                nhead=cfg.nhead, nlayers=cfg.nlayers,
                time_dim=cfg.time_dim, edge_head_hidden=cfg.edge_head_hidden
            )
            return EdgeCompetitionWrapper(
                base_node_encoder=base,
                d_edge=2*cfg.d_model,
                heads=cfg.edge_mha_heads,
                layers=cfg.edge_mha_layers,
                predict_residual=bool(cfg.predict_residual)
            )
        elif cfg.arch == "temporal":
            # temporal prior fusion wraps something; choose baseline inner by default
            inner = NodeTransformerEdgeHead(
                coord_dim=cfg.coord_dim, patch_dim=patch_dim,
                node_emb_dim=cfg.node_patch_emb_dim, d_model=cfg.d_model,
                nhead=cfg.nhead, nlayers=cfg.nlayers,
                time_dim=cfg.time_dim, edge_head_hidden=cfg.edge_head_hidden
            )
            return TemporalPriorFusionWrapper(
                inner_edge_scorer=inner,
                d_mem=cfg.d_model,
                heads=cfg.temporal_mha_heads,
                predict_residual=bool(cfg.predict_residual)
            )
        elif cfg.arch == "combo":
            # example: edgecomp + temporal
            base = NodeTransformerEdgeHead(
                coord_dim=cfg.coord_dim, patch_dim=patch_dim,
                node_emb_dim=cfg.node_patch_emb_dim, d_model=cfg.d_model,
                nhead=cfg.nhead, nlayers=cfg.nlayers,
                time_dim=cfg.time_dim, edge_head_hidden=cfg.edge_head_hidden
            )
            inner = EdgeCompetitionWrapper(
                base_node_encoder=base,
                d_edge=2*cfg.d_model,
                heads=cfg.edge_mha_heads,
                layers=cfg.edge_mha_layers,
                predict_residual=bool(cfg.predict_residual)
            )
            return TemporalPriorFusionWrapper(
                inner_edge_scorer=inner,
                d_mem=cfg.d_model,
                heads=cfg.temporal_mha_heads,
                predict_residual=bool(cfg.predict_residual)
            )
        elif cfg.arch == "edge_mlp":
            return EdgeMLPHead(hidden=cfg.edge_head_hidden)
        elif cfg.arch == "edge_mlp_attn":
            return EdgeTinyAttnHead(d_edge=cfg.edge_head_hidden, heads=cfg.temporal_mha_heads, predict_residual=bool(cfg.predict_residual))
        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")
     
    # LR scheduler
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=cfg.lr_sched_factor,
        patience=cfg.lr_sched_patience,
        verbose=cfg.lr_sched_verbose
    )
    '''
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_sched_factor,
        patience=cfg.lr_sched_patience, verbose=cfg.lr_sched_verbose
    )

    sched = cosine_schedule(cfg.T, cfg.s_min, cfg.s_max, device=device)

    # Resume if available
    start_epoch, global_step, history = try_resume(cfg, model, optimizer)
    
    # history must exist; make sure it has these keys (even if your script tracks more stuff)
    history.setdefault("train_loss", [])
    history.setdefault("train_acc", [])
    history.setdefault("train_ba", [])
    history.setdefault("val_loss", [])
    history.setdefault("val_acc", [])
    history.setdefault("val_ba", [])
    
    # Add epoch-0 baseline ONCE per run folder (works even if resuming)
    maybe_add_epoch0_baseline(cfg, model_factory, history, train_paths, val_paths, device)

    # Training
    model.train()
    t0 = time.time()
    for epoch in range(start_epoch+1, cfg.epochs+1):
        model.train()
        running_loss = 0.0; running_acc = 0.0; running_ba = 0.0; denom = 0

        # Curriculum: min_t decreases over epochs
        if cfg.use_curriculum:
            frac = (epoch-1) / max(cfg.epochs-1, 1)
            min_t = int(round(cfg.start_min_t*(1-frac) + cfg.end_min_t*frac))
            min_t = max(1, min(cfg.T, min_t))
        else:
            min_t = 1

        for batch in train_loader:
            # pre-filter valid samples so we can average loss over the batch
            valid_samples = []
            for sample in batch:
                if sample["p0"] is None: 
                    continue
                p0 = sample["p0"]; ei = sample["edge_index"]
                if p0.numel() != ei.size(1):
                    continue
                valid_samples.append(sample)

            if len(valid_samples) == 0:
                continue

            optimizer.zero_grad()
            batch_loss = 0.0; batch_acc = 0.0; batch_ba = 0.0

            for sample in valid_samples:
                p0 = sample["p0"]; ei = sample["edge_index"]; x_all = sample["x_all"]
                E_edges = ei.size(1)

                pos = max(int((p0>0.5).sum().item()), 1)
                neg = E_edges - pos
                pos_weight = float(neg/pos)

                # Edge feats + prior
                w, d_z, s_01 = compute_prior_and_edgefeats(sample)

                use_free_run = (random.random() < cfg.free_run_prob)
                if use_free_run:
                    # start from the SAME init as test
                    p_in = build_train_init(sample, w).detach()
                    # tiny unroll to teach the operator used at inference
                    steps_sched = make_inference_schedule(cfg.T, steps=cfg.free_run_steps)
                    losses = []
                    for t_step in steps_sched:
                        # augment p_in just like (A) to avoid copying
                        if cfg.p_in_dropout > 0:
                            gate = (torch.rand_like(p_in) > cfg.p_in_dropout).float()
                            p_in = gate * p_in + (1.0 - gate) * w
                        if cfg.p_in_noise_std > 0:
                            p_in = (p_in + cfg.p_in_noise_std*torch.randn_like(p_in)).clamp(0, 1)
                
                        #pred = model(x_all, ei, p_in, int(t_step), d_z, s_01)
                        #losses.append(edge_loss(pred, p0, cfg, pos_weight))
                        # update the state like inference (no detach ⇒ backprop through 1–2 steps)
                        #p_in = ((1.0 - cfg.free_run_damping) * p_in + cfg.free_run_damping * pred).clamp(0, 1)
                        
                        pred = model(x_all, ei, p_in, int(t_step), d_z, s_01)
                        # base supervised loss
                        base = edge_loss(pred, p0, cfg, pos_weight)
                        
                        N_nodes = sample["x_all"].size(0)
                        A_soft  = build_soft_adjacency(pred, ei, N=N_nodes)
                        
                        '''
                        # loop penalty on SOFT predictions (no thresholding)
                        if cfg.loop_penalty.lower() != "none" and cfg.loop_lambda > 0:
                            if cfg.loop_penalty.lower() == "simple":
                                lp = triangle_penalty(A_soft)
                            elif cfg.loop_penalty.lower() == "complex":
                                #lp = neumann_cycle_penalty(A_soft, K=int(cfg.loop_K))
                                tau = getattr(cfg, "loop_tau", 1.1)
                                est = getattr(cfg, "loop_complex_estimator", "auto")
                                R   = int(getattr(cfg, "loop_hutch_R", 8))
                                lp  = resolvent_cycle_penalty(A_soft, tau=tau, estimator=est, R=R)
                            else:
                                lp = torch.tensor(0.0, device=pred.device)
                            base = base + cfg.loop_lambda * lp
                        '''
                            
                        # loop penalty on SOFT predictions (no thresholding)
                        if cfg.loop_penalty.lower() != "none" and cfg.loop_lambda > 0:
                            pen = torch.tensor(0.0, device=A_soft.device)
                            lp = cfg.loop_penalty.lower()
                            if lp == "simple":
                                pen = triangle_penalty(A_soft)
                            elif lp == "complex":
                                tau = getattr(cfg, "loop_tau", 1.1)
                                est = getattr(cfg, "loop_complex_estimator", "auto")
                                R   = int(getattr(cfg, "loop_hutch_R", 8))
                                pen = resolvent_cycle_penalty(A_soft, tau=tau, estimator=est, R=R)
                            elif lp == "nbt":
                                # NEW: non-backtracking Hashimoto penalty (odd-only series)
                                R = int(getattr(cfg, "loop_hutch_R", 8))
                                K = int(getattr(cfg, "loop_K", 6))
                                weighting = getattr(cfg, "loop_nbt_weighting", "sqrt")
                                N_nodes = sample["x_all"].size(0)
                                pen = nonbacktracking_loop_penalty(
                                    edge_index=ei, p_edge=pred, N_nodes=N_nodes,
                                    K=K, R=R, weighting=weighting
                                )
                            # else: keep zero
                            base = base + cfg.loop_lambda * pen
                        
                        # Optional node mask: None in free-run
                        node_mask = None
                        
                        # Endpoint penalty (independent of loop penalty)
                        if cfg.endpoint_penalty.lower() != "none" and cfg.endpoint_lambda > 0:
                            if cfg.endpoint_penalty.lower() == "bump":
                                ep = endpoint_bump_penalty(A_soft, sigma=cfg.endpoint_sigma, node_mask=node_mask)
                            elif cfg.endpoint_penalty.lower() == "deg2":
                                ep = degree_to_two_penalty(A_soft, node_mask=node_mask)
                            elif cfg.endpoint_penalty.lower() == "quad":
                                ep = -quadratic_edge_penalty(A_soft)
                            else:
                                ep = torch.tensor(0.0, device=A_soft.device)
                            base = base + cfg.endpoint_lambda * ep
                        
                        losses.append(base)
                        
                        # update the state like inference (no detach ⇒ backprop through 1–2 steps)
                        p_in = ((1.0 - cfg.free_run_damping) * p_in + cfg.free_run_damping * pred).clamp(0, 1)
                
                    loss = torch.stack(losses).mean()
                else:
                    # your original teacher-forced path, but with (A) augmentation
                    t = int(torch.randint(low=min_t, high=cfg.T+1, size=(1,), device=device).item())
                    p_t = build_p_t_from_sched(p0, w, sched, t, cfg.jitter_std)
                    if cfg.p_in_dropout > 0:
                        gate = (torch.rand_like(p_t) > cfg.p_in_dropout).float()
                        p_t = gate * p_t + (1.0 - gate) * w
                    if cfg.p_in_noise_std > 0:
                        p_t = (p_t + cfg.p_in_noise_std*torch.randn_like(p_t)).clamp(0, 1)
                    #pred = model(x_all, ei, p_t, t, d_z, s_01)
                    #loss = edge_loss(pred, p0, cfg, pos_weight)
                    pred = model(x_all, ei, p_t, t, d_z, s_01)
                    # base supervised loss
                    loss = edge_loss(pred, p0, cfg, pos_weight)
                    
                    N_nodes = sample["x_all"].size(0)
                    A_soft = build_soft_adjacency(pred, ei, N=N_nodes)
                    
                    # loop penalty on SOFT predictions (no thresholding)
                    if cfg.loop_penalty.lower() != "none" and cfg.loop_lambda > 0:
                        if cfg.loop_penalty.lower() == "simple":
                            lp = triangle_penalty(A_soft)
                        elif cfg.loop_penalty.lower() == "complex":
                            #lp = neumann_cycle_penalty(A_soft, K=int(cfg.loop_K))
                            tau = getattr(cfg, "loop_tau", 1.1)
                            est = getattr(cfg, "loop_complex_estimator", "auto")
                            R   = int(getattr(cfg, "loop_hutch_R", 8))
                            lp = resolvent_cycle_penalty(A_soft, tau=tau, estimator=est, R=R)
                        elif cfg.loop_penalty.lower() == "nbt":
                            # NEW: non-backtracking Hashimoto penalty (odd-only series)
                            R = int(getattr(cfg, "loop_hutch_R", 8))
                            K = int(getattr(cfg, "loop_K", 6))
                            weighting = getattr(cfg, "loop_nbt_weighting", "sqrt")
                            N_nodes = sample["x_all"].size(0)
                            lp = nonbacktracking_loop_penalty(
                                edge_index=ei, p_edge=pred, N_nodes=N_nodes,
                                K=K, R=R, weighting=weighting
                            )
                        else:
                            lp = torch.tensor(0.0, device=pred.device)
                        loss = loss + cfg.loop_lambda * lp
                    
                    node_mask = None
                    if cfg.endpoint_mask_gt and sample.get("p0", None) is not None:
                        y_bin = (p0 > 0.5).long()
                        node_mask = gt_node_mask_from_edges(ei, y_bin, N_nodes)
                    
                    if cfg.endpoint_penalty.lower() != "none" and cfg.endpoint_lambda > 0:
                        if cfg.endpoint_penalty.lower() == "bump":
                            ep = endpoint_bump_penalty(A_soft, sigma=cfg.endpoint_sigma, node_mask=node_mask)
                        elif cfg.endpoint_penalty.lower() == "deg2":
                            ep = degree_to_two_penalty(A_soft, node_mask=node_mask)
                        elif cfg.endpoint_penalty.lower() == "quad":
                            ep = -quadratic_edge_penalty(A_soft)
                        else:
                            ep = torch.tensor(0.0, device=A_soft.device)
                        loss = loss + cfg.endpoint_lambda * ep
                        
                # average gradients across the batch (scale each sample loss)
                (loss / len(valid_samples)).backward()

                # train metrics (thresholded)
                acc, ba, _, _ = accuracy_balacc(pred.detach(), (p0>0.5).long(), cfg.thresh)
                batch_loss += loss.item(); batch_acc += acc; batch_ba += ba

            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += (batch_loss/len(valid_samples))
            running_acc  += (batch_acc/len(valid_samples))
            running_ba   += (batch_ba/len(valid_samples))
            denom += 1
            global_step += 1

            # Periodic checkpoint (latest)
            if (global_step % cfg.save_every_steps) == 0:
                save_checkpoint(cfg.out_dir, cfg, model, optimizer, epoch, global_step, history)

        # ---- End epoch: log + history ----
        train_loss = running_loss / max(denom,1)
        train_acc  = running_acc  / max(denom,1)
        train_ba   = running_ba   / max(denom,1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_ba"].append(train_ba)
        
        # Validation from max-noise (rollout you care about)
        model.eval()
        val_loss, val_acc, val_ba, n_val = eval_from_maxnoise(
            model, val_loader, cfg,
            steps=(cfg.val_infer_steps or cfg.T),
            damping=cfg.val_damping,
            subset_frac=1.0, max_batches=None, seed=cfg.seed or 1234
        )
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ba"].append(val_ba)
        
        # Train-eval from max-noise (same protocol on train set/subset)
        tfm_loss, tfm_acc, tfm_ba, n_tr = eval_from_maxnoise(
            model, train_loader, cfg,
            steps=(cfg.val_infer_steps or cfg.T),
            damping=cfg.val_damping,
            subset_frac=cfg.train_eval_frac,
            max_batches=cfg.train_eval_max_batches,
            seed=(cfg.seed or 1234) + epoch
        )
        history["train_from_max_loss"].append(tfm_loss)
        history["train_from_max_acc"].append(tfm_acc)
        history["train_from_max_ba"].append(tfm_ba)
        
        # Save plots (9 curves total)
        fig_dir2 = os.path.join(cfg.out_dir, "figs")
        dash_png = os.path.join(fig_dir2, f"dashboard_epoch_{epoch:03d}.png")
        plot_metrics_dashboard(history, dash_png, title=f"Dashboard @ epoch {epoch}")
        
        qual_png = os.path.join(fig_dir2, f"qual_epoch_{epoch:03d}.png")
        # axis="z" gives an XY max projection; change to "y" or "x" if you prefer other planes
        save_val_rollout_panel(model, val_loader, cfg, qual_png, axis="z")
        
        dash_combined = os.path.join(cfg.out_dir, "figs", f"dashboard_combined_epoch_{epoch:03d}.png")
        plot_dashboard_combined_3(history, dash_combined, title=f"Graph Diffusion Combined Dashboard @ epoch {epoch}")
        
        dash_last10 = os.path.join(cfg.out_dir, "figs", f"dashboard_last10_epoch_{epoch:03d}.png")
        plot_dashboard_last10_3(history, dash_last10, title=f"Graph Diffusion Last-10 Dashboard @ epoch {epoch}", n_last=10)

        
        # NEW: trace-style overlay (pred vs GT) using PlotAM-style 3D lines
        qual_trace_png = os.path.join(fig_dir2, f"qual_epoch_trace_{epoch:03d}.png")
        save_val_rollout_trace_panel(
            model, val_loader, cfg, qual_trace_png,
            color_pred="tab:blue",   # predicted subgraph color
            color_gt="tab:orange"    # ground-truth subgraph color
        )
        
        save_val_aug1_prob_mats(model, cfg, epoch)

        
        # LR scheduler: track what matters (val_ba)
        # metric_for_scheduler = 0.0 if np.isnan(val_ba) else val_ba
        # scheduler.step(metric_for_scheduler)
        
        metric_for_scheduler = float(val_loss if not np.isnan(val_loss) else 1e9)
        scheduler.step(metric_for_scheduler)
        
        # Checkpointing
        improved = (not np.isnan(val_ba)) and (len(history["val_ba"]) == 1 or val_ba >= max(history["val_ba"][:-1]))
        if improved:
            save_checkpoint(cfg.out_dir, cfg, model, optimizer, epoch, global_step, history, fname="best_model.pt")
        save_checkpoint(cfg.out_dir, cfg, model, optimizer, epoch, global_step, history)
        
        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} BA {train_ba:.3f} | "
            f"val (from max) loss {val_loss:.4f} acc {val_acc:.3f} BA {val_ba:.3f} | "
            f"train-eval (from max) loss {tfm_loss:.4f} acc {tfm_acc:.3f} BA {tfm_ba:.3f}"
        )
        

    print("Training complete.")
    return history

# ------------------------------ Main ----------------------------

if __name__ == "__main__":
    cli_cfg = make_cfg_from_cli()
    seed_everything(int(cli_cfg.seed) if cli_cfg.seed is not None else 42)
    history = train(cli_cfg)

