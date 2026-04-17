#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple GNN adjacency prediction from MATLAB .mat subgraphs.

Data assumptions per .mat:
  - F: (N, 3+P)   where first 3 columns are xyz coords, remaining are patch/intensity features
  - AM: (N, N)    adjacency (0/1 or logical)

We build a proposal graph using kNN on coords, run a GraphSAGE-style GNN,
and predict edge probabilities on the proposal edges to match AM on those edges.

Outputs:
  - dashboard PNGs (6 plots: train/val loss, acc, BA)
  - qualitative trace PNG (GT and Pred, non-rasterized)
  - per-epoch folder of .mat predictions for ALL validation files matching "aug1" (not aug10, etc.)
"""

from __future__ import annotations
import os, re, math, json, time, glob, random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

try:
    import scipy.io as sio
except Exception:
    sio = None


# ----------------------------- Helpers -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _bool(s):
    if isinstance(s, bool): return s
    s = str(s).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_plot_style_ar12():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

def list_mat_files(root: str, subdir: str) -> List[str]:
    base = os.path.join(root, subdir)
    if not os.path.isdir(base):
        return []
    paths = []
    for r, _, fs in os.walk(base):
        for f in fs:
            if f.lower().endswith(".mat"):
                paths.append(os.path.join(r, f))
    paths.sort()
    return paths

def is_aug1_file(path: str) -> bool:
    # matches 'aug1' but not 'aug10', 'aug11', etc.
    name = os.path.basename(path).lower()
    return re.search(r"aug1(?!\d)", name) is not None

def safe_loadmat(path: str) -> Dict[str, Any]:
    if sio is None:
        raise RuntimeError("scipy is required: pip install scipy")
    d = sio.loadmat(path)
    # strip meta keys
    return {k: v for k, v in d.items() if not k.startswith("__")}

def to_numpy_2d(x) -> np.ndarray:
    """Handle possible MATLAB weirdness (nested arrays)."""
    if isinstance(x, np.ndarray):
        # If it's an object array with one element, unwrap
        if x.dtype == object and x.size == 1:
            return np.array(x.item())
        return np.array(x)
    return np.array(x)
    
def mat_to_dense_np(x) -> np.ndarray:
    """
    Convert MATLAB-loaded arrays to a clean dense numpy array.
    Handles:
      - numpy arrays
      - 1-element object arrays (common loadmat cell wrapping)
      - scipy sparse matrices (e.g., coo_matrix)
    """
    # Unwrap 1-element object arrays (MATLAB cells)
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()

    # Convert scipy sparse -> dense
    try:
        import scipy.sparse as sp
        if sp.issparse(x):
            x = x.toarray()
    except Exception:
        # fallback: some sparse-like objects expose toarray()
        if hasattr(x, "toarray"):
            x = x.toarray()

    # Finally ensure ndarray
    return np.asarray(x)


def zscore_coords(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = coords.mean(dim=0, keepdim=True)
    s = coords.std(dim=0, keepdim=True)
    return (coords - m) / (s + eps)

def build_knn_edges(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    coords: (N,3) float tensor on CPU recommended
    returns undirected edge_index (2,E) with i<j unique pairs
    """
    assert coords.dim() == 2 and coords.size(1) == 3
    N = coords.size(0)
    if N <= 1:
        return torch.zeros((2,0), dtype=torch.long)

    kk = int(min(k, max(N-1, 1)))
    # pairwise distances
    D = torch.cdist(coords, coords, p=2)  # (N,N)
    D.fill_diagonal_(float("inf"))
    # nearest neighbors
    nn_idx = torch.topk(-D, k=kk, dim=1).indices  # (N,kk) using negative distance

    src = torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1, kk).reshape(-1)
    dst = nn_idx.reshape(-1)

    a = torch.minimum(src, dst)
    b = torch.maximum(src, dst)
    pairs = torch.stack([a, b], dim=1)  # (N*kk,2)
    pairs = pairs[a != b]

    # unique rows
    pairs = torch.unique(pairs, dim=0)
    if pairs.numel() == 0:
        return torch.zeros((2,0), dtype=torch.long)

    edge_index = pairs.t().contiguous()  # (2,E), i<j
    return edge_index

def make_directed_with_self_loops(edge_und: torch.Tensor, N: int, device: torch.device) -> torch.Tensor:
    """
    edge_und: (2,E) with i<j
    returns directed edges (2, 2E + N) including both directions + self loops
    """
    if edge_und.numel() == 0:
        loops = torch.arange(N, device=device, dtype=torch.long)
        return torch.stack([loops, loops], dim=0)

    e = edge_und.to(device)
    rev = torch.stack([e[1], e[0]], dim=0)
    loops = torch.arange(N, device=device, dtype=torch.long)
    self_e = torch.stack([loops, loops], dim=0)
    return torch.cat([e, rev, self_e], dim=1)

def edge_labels_from_AM(edge_und: torch.Tensor, AM: torch.Tensor) -> torch.Tensor:
    """
    edge_und: (2,E) i<j
    AM: (N,N) 0/1
    returns y: (E,) float in {0,1}
    """
    if edge_und.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=AM.device)
    i, j = edge_und[0], edge_und[1]
    y = AM[i, j].float()
    return y.clamp(0, 1)

def accuracy_balacc_from_logits(logits: torch.Tensor, y: torch.Tensor, thresh: float) -> Tuple[float, float]:
    p = torch.sigmoid(logits)
    yhat = (p >= thresh).long()
    yb = (y > 0.5).long()

    tp = int(((yhat == 1) & (yb == 1)).sum().item())
    tn = int(((yhat == 0) & (yb == 0)).sum().item())
    fp = int(((yhat == 1) & (yb == 0)).sum().item())
    fn = int(((yhat == 0) & (yb == 1)).sum().item())

    tot = max(int(y.numel()), 1)
    acc = (tp + tn) / tot
    tpr = tp / max(tp + fn, 1e-9)
    tnr = tn / max(tn + fp, 1e-9)
    ba = 0.5 * (tpr + tnr)
    return float(acc), float(ba)

def build_soft_adjacency_from_edges(edge_und: torch.Tensor, p_edge: torch.Tensor, N: int, device: torch.device) -> torch.Tensor:
    """
    Create dense NxN soft adjacency from undirected edges and probabilities.
    """
    A = torch.zeros((N, N), dtype=p_edge.dtype, device=device)
    if edge_und.numel() == 0:
        return A
    i, j = edge_und[0].to(device), edge_und[1].to(device)
    w = p_edge.to(device)
    A.index_put_((i, j), w, accumulate=False)
    A.index_put_((j, i), w, accumulate=False)
    A.fill_diagonal_(0.0)
    return A

def A_mv_from_edge_list(edge_dir: torch.Tensor, w_dir: torch.Tensor, x: torch.Tensor, N: int) -> torch.Tensor:
    """
    Compute y = A x using directed edge list.
      edge_dir: (2,Ed) with src->dst
      w_dir: (Ed,) weights
      x: (N,) vector
    """
    src, dst = edge_dir[0], edge_dir[1]
    msg = w_dir * x[src]
    y = torch.zeros((N,), device=x.device, dtype=x.dtype)
    y.index_add_(0, dst, msg)
    return y

def triangle_penalty_hutch(edge_und: torch.Tensor, p_edge: torch.Tensor, N: int, R: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """
    Estimate trace(A^3)/6 via Hutchinson, using matvecs without dense NxN matmul.
    A is symmetric with weights on undirected edges.
    """
    if edge_und.numel() == 0 or p_edge.numel() == 0:
        return torch.zeros((), device=p_edge.device, dtype=p_edge.dtype)

    device = p_edge.device
    dtype = p_edge.dtype

    # build directed edges and directed weights (both directions)
    i, j = edge_und[0].to(device), edge_und[1].to(device)
    w = p_edge.to(device).clamp(0, 1)
    edge_dir = torch.cat([torch.stack([i, j], dim=0),
                          torch.stack([j, i], dim=0)], dim=1)  # (2, 2E)
    w_dir = torch.cat([w, w], dim=0)  # (2E,)

    acc = torch.zeros((), device=device, dtype=dtype)
    for _ in range(int(R)):
        z = torch.empty((N,), device=device, dtype=dtype).bernoulli_().mul_(2).add_(-1)  # ±1
        y1 = A_mv_from_edge_list(edge_dir, w_dir, z, N)
        y2 = A_mv_from_edge_list(edge_dir, w_dir, y1, N)
        y3 = A_mv_from_edge_list(edge_dir, w_dir, y2, N)
        acc = acc + (z * y3).sum()

    trA3 = acc / float(R)
    tri_mass = trA3 / 6.0

    E_est = w.sum()  # undirected edge mass
    denom = (N + E_est + eps)
    return tri_mass / denom


# ------------------------- Plotting traces -------------------------

def _set_axes_equal_3d(ax):
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

def plot_am_py(AM: np.ndarray, r: np.ndarray, color: str, ax, linewidth: float = 1.2):
    AM = np.maximum(AM, AM.T)
    AMu = np.triu(AM)
    ii, jj = np.nonzero(AMu)
    for a, b in zip(ii, jj):
        X = [r[a, 0], r[b, 0]]
        Y = [r[a, 1], r[b, 1]]
        Z = [r[a, 2], r[b, 2]]
        ax.plot(Y, X, Z, color=color, linewidth=linewidth)

def save_val_trace_topdown(gt_AM: np.ndarray,
                           pred_A: np.ndarray,
                           r_xyz: np.ndarray,
                           out_png: str,
                           title: str):
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_am_py(gt_AM, r_xyz, color="tab:blue", ax=ax1, linewidth=1.5)
    _set_axes_equal_3d(ax1)
    ax1.view_init(elev=90, azim=-90)
    ax1.set_xlabel("Y"); ax1.set_ylabel("X")
    ax1.set_zlabel(""); ax1.set_zticks([])
    ax1.grid(False)
    ax1.set_title("Ground truth (XY top view)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_am_py(pred_A, r_xyz, color="tab:blue", ax=ax2, linewidth=1.5)
    _set_axes_equal_3d(ax2)
    ax2.view_init(elev=90, azim=-90)
    ax2.set_xlabel("Y"); ax2.set_ylabel("X")
    ax2.set_zlabel(""); ax2.set_zticks([])
    ax2.grid(False)
    ax2.set_title("Prediction (XY top view)")

    fig.suptitle(title, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------- Model -----------------------------

class PatchMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, out_dim), nn.ReLU(), nn.Dropout(0.2),
        )
    def forward(self, x): return self.net(x)

class GraphSAGELayer(nn.Module):
    """
    Mean aggregator GraphSAGE in pure PyTorch using index_add_.
    """
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_nei  = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_dir: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        src, dst = edge_dir[0], edge_dir[1]
        msg = h[src]  # (Ed, d_in)

        agg = torch.zeros((N, h.size(1)), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros((N,), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones((dst.numel(),), device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(1)

        out = self.lin_self(h) + self.lin_nei(agg)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class EdgeMLP(nn.Module):
    def __init__(self, d_node: int, hidden: int = 256):
        super().__init__()
        # features: [hi, hj, |hi-hj|, delta_xyz(3)] => 3*d_node + 3
        self.net = nn.Sequential(
            nn.Linear(3*d_node + 3, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1)  # logits
        )

    def forward(self, h: torch.Tensor, edge_und: torch.Tensor, coords_z: torch.Tensor) -> torch.Tensor:
        if edge_und.numel() == 0:
            return torch.zeros((0,), device=h.device, dtype=h.dtype)
        i, j = edge_und[0], edge_und[1]
        hi, hj = h[i], h[j]
        delta = coords_z[i] - coords_z[j]
        feats = torch.cat([hi, hj, (hi - hj).abs(), delta], dim=1)
        return self.net(feats).view(-1)

class SimpleAdjGNN(nn.Module):
    def __init__(self, patch_dim: int, hidden_dim: int, nlayers: int, dropout: float):
        super().__init__()
        self.use_patch = patch_dim > 0
        self.patch_enc = PatchMLP(patch_dim, out_dim=64) if self.use_patch else None

        in_dim = 3 + (64 if self.use_patch else 0)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        self.layers = nn.ModuleList([
            GraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(int(nlayers))
        ])

        self.edge_head = EdgeMLP(hidden_dim, hidden=256)

    def forward(self, F_all: torch.Tensor, edge_und: torch.Tensor, edge_dir: torch.Tensor) -> torch.Tensor:
        coords = F_all[:, :3]
        coords_z = zscore_coords(coords)

        feats = [coords_z]
        if self.use_patch:
            feats.append(self.patch_enc(F_all[:, 3:]))

        h = self.in_proj(torch.cat(feats, dim=1))
        for layer in self.layers:
            h = layer(h, edge_dir)

        logits = self.edge_head(h, edge_und, coords_z)
        return logits


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    # data
    data_root: str = ""
    train_subdir: str = "train"
    val_subdir: str = "validation"

    # run io
    out_dir: str = "outputs_gnn_adj"
    run_name: Optional[str] = None
    resume_if_found: bool = True
    ckpt_name: str = "checkpoint_latest.pt"

    # training
    epochs: int = 30
    lr: float = 1e-3
    grad_clip: float = 1.0
    thresh: float = 0.5

    # model
    hidden_dim: int = 128
    nlayers: int = 2
    dropout: float = 0.1
    knn_k: int = 16

    # loop penalty (triangles)
    loop_penalty: bool = False
    loop_lambda: float = 0.0
    loop_hutch_R: int = 8

    # misc
    seed: int = 42
    use_gpu: bool = True


# ----------------------------- CLI -----------------------------

def make_cfg_from_cli() -> Config:
    import argparse
    from datetime import datetime

    p = argparse.ArgumentParser("Train simple GNN adjacency predictor from .mat files")

    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--train-subdir", type=str, default=None)
    p.add_argument("--val-subdir", type=str, default=None)

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--thresh", type=float, default=None)

    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--nlayers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--knn-k", type=int, default=None)

    p.add_argument("--loop-penalty", type=_bool, default=None)
    p.add_argument("--loop-lambda", type=float, default=None)
    p.add_argument("--loop-hutch-R", type=int, default=None)

    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume-if-found", type=_bool, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use-gpu", type=_bool, default=None)

    args = p.parse_args()
    cfg = Config()
    cfg.data_root = args.data_root

    # apply overrides
    for k, v in vars(args).items():
        if v is None: continue
        if hasattr(cfg, k.replace("-", "_")):
            setattr(cfg, k.replace("-", "_"), v)

    if args.train_subdir is not None: cfg.train_subdir = args.train_subdir
    if args.val_subdir is not None: cfg.val_subdir = args.val_subdir

    # resolve run folder like your diffusion script: OUT/run_name_timestamp, resume to newest match
    base_out = args.out_dir or cfg.out_dir
    run_name = args.run_name

    if run_name:
        pattern = os.path.join(base_out, f"{run_name}_*")
        matches = sorted(glob.glob(pattern))
        if cfg.resume_if_found and matches:
            cfg.out_dir = matches[-1]
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg.out_dir = os.path.join(base_out, f"{run_name}_{ts}")
    else:
        cfg.out_dir = base_out

    ensure_dir(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "config_resolved.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return cfg


# -------------------------- Checkpointing --------------------------

def save_checkpoint(cfg: Config, model: nn.Module, opt: torch.optim.Optimizer,
                    epoch: int, history: Dict[str, List[float]], fname: Optional[str] = None):
    ensure_dir(cfg.out_dir)
    ck = {
        "cfg": asdict(cfg),
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        "history": history,
    }
    torch.save(ck, os.path.join(cfg.out_dir, fname or cfg.ckpt_name))

def try_resume(cfg: Config, model: nn.Module, opt: torch.optim.Optimizer):
    path = os.path.join(cfg.out_dir, cfg.ckpt_name)
    if cfg.resume_if_found and os.path.exists(path):
        ck = torch.load(path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        epoch = int(ck.get("epoch", 0))
        hist = ck.get("history", None)
        if hist is None:
            hist = {"train_loss": [], "train_acc": [], "train_ba": [],
                    "val_loss": [], "val_acc": [], "val_ba": []}
        print(f"[resume] Loaded {path} at epoch={epoch}")
        return epoch, hist
    return 0, {"train_loss": [], "train_acc": [], "train_ba": [],
               "val_loss": [], "val_acc": [], "val_ba": []}


# ---------------------------- Dashboard ----------------------------

def plot_dashboard_6(history: Dict[str, List[float]], out_png: str, title: str):
    layout = [
        ["train_loss", "train_acc", "train_ba"],
        ["val_loss", "val_acc", "val_ba"],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for r in range(2):
        for c in range(3):
            k = layout[r][c]
            ax = axes[r, c]
            ys = history.get(k, [])
            if len(ys) > 0 and np.isfinite(np.array(ys, dtype=float)).all():
                ax.plot(ys, marker="o")
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
            ax.set_title(k)
            ax.set_xlabel("epoch")
            ax.grid(False)
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    
def plot_dashboard_combined_3(history: Dict[str, List[float]], out_png: str, title: str):
    """
    3-panel dashboard:
      - Loss
      - Accuracy
      - Balanced Accuracy

    Train = blue, Val = orange
    Lines only (no markers)
    Accuracy/BA y-limits fixed to [0, 1]
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

        # epochs are 1..len, but if you prepend a "pre-train" point, that's epoch 0.
        x_tr = np.arange(len(y_tr))
        x_va = np.arange(len(y_va))

        plotted_any = False
        if len(y_tr) > 0 and np.isfinite(y_tr).any():
            ax.plot(x_tr, y_tr, linewidth=2.0, label="train")  # default color = blue
            plotted_any = True

        if len(y_va) > 0 and np.isfinite(y_va).any():
            ax.plot(x_va, y_va, linewidth=2.0, label="val")    # default color = orange
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")

        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(False)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)

        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _slice_recent(y: np.ndarray, n_recent: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for the last n_recent points. x is 0..len(y_recent)-1."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.arange(0), y
    if n_recent is None or n_recent <= 0 or y.size <= n_recent:
        y_recent = y
    else:
        y_recent = y[-n_recent:]
    x = np.arange(y_recent.size)
    return x, y_recent


def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    """Simple centered-ish moving average (actually trailing to avoid look-ahead)."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    win = int(max(win, 1))
    if y.size < win:
        # if too short, just return mean as a flat line
        return np.full_like(y, np.nanmean(y))
    # trailing MA
    kernel = np.ones(win, dtype=float) / float(win)
    y_ma = np.convolve(y, kernel, mode="valid")
    # pad to original length (left pad with first valid value)
    pad = y.size - y_ma.size
    return np.concatenate([np.full(pad, y_ma[0]), y_ma])


def _ema(y: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Exponential moving average; alpha in (0,1], higher = less smoothing."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    alpha = float(alpha)
    alpha = min(max(alpha, 1e-6), 1.0)
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, y.size):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def plot_dashboard_recent_3(history: Dict[str, List[float]],
                            out_png: str,
                            title: str,
                            n_recent: int = 75):
    """
    Recent-window 3-panel dashboard (last n_recent epochs):
      - Loss
      - Accuracy
      - Balanced Accuracy

    Train = blue, Val = orange (matplotlib defaults via call order)
    Lines only, no markers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("loss", "train_loss", "val_loss", None),
        ("acc",  "train_acc",  "val_acc",  (8.0, 1.0)),
        ("ba",   "train_ba",   "val_ba",   (8.0, 1.0)),
    ]

    for ax, (name, k_tr, k_va, ylim) in zip(axes, panels):
        y_tr_full = np.asarray(history.get(k_tr, []), dtype=float)
        y_va_full = np.asarray(history.get(k_va, []), dtype=float)

        x_tr, y_tr = _slice_recent(y_tr_full, n_recent)
        x_va, y_va = _slice_recent(y_va_full, n_recent)

        plotted_any = False
        if y_tr.size > 0 and np.isfinite(y_tr).any():
            ax.plot(x_tr, y_tr, linewidth=2.0, label="train")  # blue
            plotted_any = True
        if y_va.size > 0 and np.isfinite(y_va).any():
            ax.plot(x_va, y_va, linewidth=2.0, label="val")    # orange
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")

        ax.set_title(f"{name} (last {min(n_recent, max(len(y_tr_full), len(y_va_full)))} epochs)")
        ax.set_xlabel("recent epoch index")
        ax.grid(False)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)

        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_dashboard_recent_smoothed_3(history: Dict[str, List[float]],
                                     out_png: str,
                                     title: str,
                                     n_recent: int = 75,
                                     method: str = "ema",
                                     ma_win: int = 9,
                                     ema_alpha: float = 0.2):
    """
    Recent-window 3-panel dashboard with smoothing.
    Smoothing is applied AFTER slicing to last n_recent points.

    method:
      - "ma": moving average with window ma_win
      - "ema": exponential moving average with alpha ema_alpha
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("loss", "train_loss", "val_loss", None),
        ("acc",  "train_acc",  "val_acc",  (0.8, 1.0)),
        ("ba",   "train_ba",   "val_ba",   (0.8, 1.0)),
    ]

    method = str(method).lower().strip()

    def smooth(y: np.ndarray) -> np.ndarray:
        if y.size == 0:
            return y
        if method == "ma":
            return _moving_average(y, ma_win)
        # default to EMA
        return _ema(y, ema_alpha)

    for ax, (name, k_tr, k_va, ylim) in zip(axes, panels):
        y_tr_full = np.asarray(history.get(k_tr, []), dtype=float)
        y_va_full = np.asarray(history.get(k_va, []), dtype=float)

        x_tr, y_tr = _slice_recent(y_tr_full, n_recent)
        x_va, y_va = _slice_recent(y_va_full, n_recent)

        y_tr_s = smooth(y_tr)
        y_va_s = smooth(y_va)

        plotted_any = False
        if y_tr_s.size > 0 and np.isfinite(y_tr_s).any():
            ax.plot(x_tr, y_tr_s, linewidth=2.0, label="train (smoothed)")  # blue
            plotted_any = True
        if y_va_s.size > 0 and np.isfinite(y_va_s).any():
            ax.plot(x_va, y_va_s, linewidth=2.0, label="val (smoothed)")    # orange
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")

        ax.set_title(f"{name} smoothed ({method}), last {min(n_recent, max(len(y_tr_full), len(y_va_full)))}")
        ax.set_xlabel("recent epoch index")
        ax.grid(False)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)

        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_dashboard_first80_loss_ba(history: Dict[str, List[float]],
                                   out_png: str,
                                   title: str,
                                   n_first: int = 80):
    """
    2-panel dashboard, first n_first epochs only.
      - Loss
      - Balanced Accuracy

    Train and Val with the same styling as plot_dashboard_combined_3.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    panels = [
        ("loss", "train_loss", "val_loss", None),
        ("ba",   "train_ba",   "val_ba",   (0.0, 1.0)),
    ]

    for ax, (name, k_tr, k_va, ylim) in zip(axes, panels):
        y_tr_full = np.asarray(history.get(k_tr, []), dtype=float)
        y_va_full = np.asarray(history.get(k_va, []), dtype=float)

        y_tr = y_tr_full[:n_first]
        y_va = y_va_full[:n_first]

        x_tr = np.arange(len(y_tr))
        x_va = np.arange(len(y_va))

        plotted_any = False
        if len(y_tr) > 0 and np.isfinite(y_tr).any():
            ax.plot(x_tr, y_tr, linewidth=2.0, label="train")
            plotted_any = True

        if len(y_va) > 0 and np.isfinite(y_va).any():
            ax.plot(x_va, y_va, linewidth=2.0, label="val")
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")

        ax.set_title(f"{name} (first {min(n_first, max(len(y_tr_full), len(y_va_full)))} epochs)")
        ax.set_xlabel("epoch")
        ax.grid(False)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)

        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------- Data read ----------------------------

def load_graph_from_mat(path: str, device: torch.device) -> Optional[Dict[str, Any]]:
    d = safe_loadmat(path)
    if "F" not in d or "AM" not in d:
        return None

    F_np  = mat_to_dense_np(d["F"])
    AM_np = mat_to_dense_np(d["AM"])

    F_np  = np.squeeze(F_np)
    AM_np = np.squeeze(AM_np)

    # ---- skip empty graphs ----
    if F_np.size == 0 or AM_np.size == 0:
        return None

    # Ensure 2D
    if F_np.ndim != 2 or AM_np.ndim != 2:
        return None

    # Must have xyz
    if F_np.shape[1] < 3:
        return None

    N = F_np.shape[0]
    if AM_np.shape[0] != N or AM_np.shape[1] != N:
        return None

    # dtype + binarize
    F_np = F_np.astype(np.float32, copy=False)
    AM_np = (AM_np > 0.5).astype(np.float32, copy=False)

    F_t  = torch.from_numpy(F_np).to(device)
    AM_t = torch.from_numpy(AM_np).to(device)
    return {"F": F_t, "AM": AM_t, "path": path}



# ---------------------------- Eval + saving ----------------------------

@torch.no_grad()
def eval_paths(model: nn.Module, paths: List[str], cfg: Config, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    losses, accs, bas = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    for pth in paths:
        g = load_graph_from_mat(pth, device)
        if g is None:
            continue
        F_all, AM = g["F"], g["AM"]
        N = F_all.size(0)
        if N <= 1:
            continue

        # proposal edges
        edge_und = build_knn_edges(F_all[:, :3].detach().cpu(), cfg.knn_k)  # CPU
        if edge_und.numel() == 0:
            continue
        edge_und = edge_und.to(device)
        edge_dir = make_directed_with_self_loops(edge_und, N, device)

        y = edge_labels_from_AM(edge_und, AM)  # (E,)
        if y.numel() == 0:
            continue

        pos = float((y > 0.5).sum().item())
        neg = float(y.numel() - pos)
        if pos < 1:
            # no positives => balanced acc meaningless; skip
            continue
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
        bce_w = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

        logits = model(F_all, edge_und, edge_dir)
        loss = bce_w(logits, y)

        if cfg.loop_penalty and cfg.loop_lambda > 0:
            p_edge = torch.sigmoid(logits)
            lp = triangle_penalty_hutch(edge_und, p_edge, N=N, R=cfg.loop_hutch_R)
            loss = loss + cfg.loop_lambda * lp

        acc, ba = accuracy_balacc_from_logits(logits, y, cfg.thresh)
        losses.append(loss.item()); accs.append(acc); bas.append(ba)

    if len(losses) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(accs)), float(np.mean(bas))

@torch.no_grad()
def save_val_aug1_matfiles(model: nn.Module, cfg: Config, epoch: int, val_aug1_paths: List[str], device: torch.device) -> int:
    if sio is None:
        raise RuntimeError("scipy is required to save .mat files (scipy.io.savemat).")

    out_folder = os.path.join(cfg.out_dir, f"validation_aug1_epoch{epoch:03d}")
    ensure_dir(out_folder)

    model.eval()
    saved = 0
    for pth in val_aug1_paths:
        g = load_graph_from_mat(pth, device)
        if g is None:
            continue
        F_all, AM = g["F"], g["AM"]
        N = F_all.size(0)
        if N <= 1:
            continue

        edge_und = build_knn_edges(F_all[:, :3].detach().cpu(), cfg.knn_k)
        if edge_und.numel() == 0:
            continue
        edge_und = edge_und.to(device)
        edge_dir = make_directed_with_self_loops(edge_und, N, device)

        logits = model(F_all, edge_und, edge_dir)
        p_edge = torch.sigmoid(logits)

        A_prob = build_soft_adjacency_from_edges(edge_und, p_edge, N=N, device=device)

        coords = F_all[:, :3]
        base = os.path.splitext(os.path.basename(pth))[0]
        mat_path = os.path.join(out_folder, base + ".mat")

        sio.savemat(
            mat_path,
            {
                "A_prob": A_prob.detach().cpu().numpy(),
                "AM_gt": AM.detach().cpu().numpy(),
                "r": coords.detach().cpu().numpy(),
                "p_edge": p_edge.detach().cpu().numpy(),
                "edge_index": edge_und.detach().cpu().numpy(),
                "source_path": pth,
            },
            do_compression=True
        )
        saved += 1

    print(f"[val_aug1] Saved {saved} .mat files to: {out_folder}")
    return saved

@torch.no_grad()
def save_one_val_trace_png(model: nn.Module, cfg: Config, epoch: int,
                           val_aug1_paths: List[str], device: torch.device):
    # pick first non-empty
    pick = None
    for pth in val_aug1_paths:
        g = load_graph_from_mat(pth, device)
        if g is None:
            continue
        AM = g["AM"]
        if AM.numel() > 0 and AM.sum().item() > 0:
            pick = pth
            break
    if pick is None:
        return

    g = load_graph_from_mat(pick, device)
    if g is None:
        return

    F_all, AM = g["F"], g["AM"]
    N = F_all.size(0)
    if N <= 1:
        return

    edge_und = build_knn_edges(F_all[:, :3].detach().cpu(), cfg.knn_k)
    if edge_und.numel() == 0:
        return
    edge_und = edge_und.to(device)
    edge_dir = make_directed_with_self_loops(edge_und, N, device)

    logits = model(F_all, edge_und, edge_dir)
    p_edge = torch.sigmoid(logits)

    # ---------------- NEW: compute single-graph BA on proposal edges ----------------
    y = edge_labels_from_AM(edge_und, AM)  # (E,)
    ba_one = float("nan")
    if y.numel() > 0:
        pos = int((y > 0.5).sum().item())
        neg = int(y.numel() - pos)
        # BA is only meaningful if both classes exist
        if pos > 0 and neg > 0:
            _, ba_one = accuracy_balacc_from_logits(logits, y, cfg.thresh)

    ba_str = f"{ba_one:.3f}" if np.isfinite(ba_one) else "n/a"
    # ------------------------------------------------------------------------------

    # threshold into a predicted adjacency (only on proposal edges)
    pred_mask = (p_edge >= cfg.thresh)
    edge_sel = edge_und[:, pred_mask]

    A_pred = torch.zeros((N, N), device=device, dtype=torch.float32)
    if edge_sel.numel() > 0:
        i, j = edge_sel[0], edge_sel[1]
        A_pred[i, j] = 1.0
        A_pred[j, i] = 1.0
    A_pred.fill_diagonal_(0.0)

    out_png = os.path.join(cfg.out_dir, "figs", f"val_trace_epoch_{epoch:03d}.png")
    title = (
        f"Val trace (aug1) @ epoch {epoch} | thresh={cfg.thresh:.2f} | BA={ba_str}\n"
        f"{os.path.basename(pick)}"
    )

    save_val_trace_topdown(
        gt_AM=AM.detach().cpu().numpy(),
        pred_A=A_pred.detach().cpu().numpy(),
        r_xyz=F_all[:, :3].detach().cpu().numpy(),
        out_png=out_png,
        title=title
    )


@torch.no_grad()
def maybe_add_epoch0_baseline(cfg: Config,
                             patch_dim: int,
                             history: Dict[str, List[float]],
                             train_paths: List[str],
                             val_aug1_paths: List[str],
                             device: torch.device):
    """
    Adds an epoch-0 baseline (init model) to history ONCE per run folder,
    even if resuming. Uses:
      - init weights saved in cfg.out_dir/init_model_state.pt (created if missing)
      - a flag file cfg.out_dir/baseline_epoch0_done.txt to avoid duplicates

    Baseline values are PREPENDED (insert at index 0) so epoch indexing becomes:
      0 = init baseline, 1.. = trained epochs
    """
    DO_PRETRAIN_BASELINE = True  # <- your toggle (easy to comment out)

    if not DO_PRETRAIN_BASELINE:
        return

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "figs"))

    flag_path = os.path.join(cfg.out_dir, "baseline_epoch0_done.txt")
    init_path = os.path.join(cfg.out_dir, "init_model_state.pt")

    # If we've already done it for this run folder, do nothing
    if os.path.exists(flag_path):
        return

    # Build a temporary model for baseline eval (does NOT affect your training model)
    model0 = SimpleAdjGNN(
        patch_dim=patch_dim,
        hidden_dim=cfg.hidden_dim,
        nlayers=cfg.nlayers,
        dropout=cfg.dropout
    ).to(device)

    # If init weights exist, load them; otherwise create + save init weights deterministically
    if os.path.exists(init_path):
        model0.load_state_dict(torch.load(init_path, map_location=device))
    else:
        # Create init weights without perturbing training RNG state
        # (so your resumed training randomness isn't changed)
        devices = [torch.device("cuda")] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            seed_everything(cfg.seed)
            model0 = SimpleAdjGNN(
                patch_dim=patch_dim,
                hidden_dim=cfg.hidden_dim,
                nlayers=cfg.nlayers,
                dropout=cfg.dropout
            ).to(device)
        torch.save(model0.state_dict(), init_path)

    print("[baseline] Computing epoch-0 (init) metrics and prepending to history...")

    tr0_loss, tr0_acc, tr0_ba = eval_paths(model0, train_paths, cfg, device)
    va0_loss, va0_acc, va0_ba = eval_paths(model0, val_aug1_paths, cfg, device)

    # Prepend (epoch 0)
    history["train_loss"].insert(0, tr0_loss)
    history["train_acc"].insert(0, tr0_acc)
    history["train_ba"].insert(0, tr0_ba)

    history["val_loss"].insert(0, va0_loss)
    history["val_acc"].insert(0, va0_acc)
    history["val_ba"].insert(0, va0_ba)

    # Save baseline combined dashboard image
    dash0_png = os.path.join(cfg.out_dir, "figs", "dashboard_combined_epoch_000.png")
    plot_dashboard_combined_3(history, dash0_png, title="GNN Adj Combined Dashboard @ epoch 0")

    # Write flag so it won't run again for this run folder
    with open(flag_path, "w") as f:
        f.write("baseline epoch0 added\n")


# ----------------------------- Train -----------------------------

def train(cfg: Config):
    device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    seed_everything(cfg.seed)
    set_plot_style_ar12()

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "figs"))

    train_paths = list_mat_files(cfg.data_root, cfg.train_subdir)
    val_paths_all = list_mat_files(cfg.data_root, cfg.val_subdir)
    val_aug1_paths = [p for p in val_paths_all if is_aug1_file(p)]

    print(f"Train .mat: {len(train_paths)}")
    print(f"Val   .mat: {len(val_paths_all)} | Val aug1: {len(val_aug1_paths)}")

    if len(train_paths) == 0:
        raise RuntimeError("No training .mat files found.")
    if len(val_aug1_paths) == 0:
        print("[warn] No aug1 validation files found; val metrics may be n/a.")

    # peek patch dim
    g0 = None
    for p in train_paths[:5000]:  # cap search to avoid worst-case full scan
        g0 = load_graph_from_mat(p, device=torch.device("cpu"))
        if g0 is not None:
            break
    if g0 is None:
        raise RuntimeError("Could not find any non-empty training graphs with valid F/AM.")
    
    F0 = g0["F"]
    
    patch_dim = max(int(F0.size(1)) - 3, 0)
    print(f"Feature dim: {F0.size(1)} (patch_dim={patch_dim})")

    model = SimpleAdjGNN(
        patch_dim=patch_dim,
        hidden_dim=cfg.hidden_dim,
        nlayers=cfg.nlayers,
        dropout=cfg.dropout
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    start_epoch, history = try_resume(cfg, model, opt)
    
    # Add epoch-0 baseline ONCE per run folder (works even if resuming)
    maybe_add_epoch0_baseline(cfg, patch_dim, history, train_paths, val_aug1_paths, device)


    t0 = time.time()
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        model.train()
        random.shuffle(train_paths)

        losses, accs, bas = [], [], []

        for pth in train_paths:
            g = load_graph_from_mat(pth, device)
            if g is None:
                continue
            F_all, AM = g["F"], g["AM"]
            
            N = F_all.size(0)
            if N <= 1:
                continue

            edge_und = build_knn_edges(F_all[:, :3].detach().cpu(), cfg.knn_k)
            if edge_und.numel() == 0:
                continue
            edge_und = edge_und.to(device)
            edge_dir = make_directed_with_self_loops(edge_und, N, device)

            y = edge_labels_from_AM(edge_und, AM)  # (E,)
            if y.numel() == 0:
                continue

            pos = float((y > 0.5).sum().item())
            neg = float(y.numel() - pos)
            if pos < 1:
                # skip pathological proposal that has zero positives
                continue
            pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

            opt.zero_grad()
            logits = model(F_all, edge_und, edge_dir)
            loss = bce(logits, y)

            if cfg.loop_penalty and cfg.loop_lambda > 0:
                p_edge = torch.sigmoid(logits)
                lp = triangle_penalty_hutch(edge_und, p_edge, N=N, R=cfg.loop_hutch_R)
                loss = loss + cfg.loop_lambda * lp

            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            acc, ba = accuracy_balacc_from_logits(logits.detach(), y.detach(), cfg.thresh)
            losses.append(loss.item()); accs.append(acc); bas.append(ba)

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc  = float(np.mean(accs)) if accs else float("nan")
        train_ba   = float(np.mean(bas)) if bas else float("nan")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_ba"].append(train_ba)

        # validation on ALL aug1
        val_loss, val_acc, val_ba = eval_paths(model, val_aug1_paths, cfg, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ba"].append(val_ba)

        # dashboard
        dash_png = os.path.join(cfg.out_dir, "figs", f"dashboard_epoch_{epoch:03d}.png")
        plot_dashboard_6(history, dash_png, title=f"GNN Adj Dashboard @ epoch {epoch}")
        
        # additional dashboard (combined 3-panel, train vs val)
        dash2_png = os.path.join(cfg.out_dir, "figs", f"dashboard_combined_epoch_{epoch:03d}.png")
        plot_dashboard_combined_3(history, dash2_png, title=f"GNN Adj Combined Dashboard @ epoch {epoch}")
        
        # recent window (last ~75 epochs)
        dash_recent = os.path.join(cfg.out_dir, "figs", f"dashboard_recent75_epoch_{epoch:03d}.png")
        plot_dashboard_recent_3(history, dash_recent, title=f"GNN Adj Recent (75) @ epoch {epoch}", n_recent=75)
        
        # recent window + smoothed
        dash_recent_s = os.path.join(cfg.out_dir, "figs", f"dashboard_recent75_smoothed_epoch_{epoch:03d}.png")
        plot_dashboard_recent_smoothed_3(
            history,
            dash_recent_s,
            title=f"GNN Adj Recent (75) Smoothed @ epoch {epoch}",
            n_recent=75,
            method="ema",      # or "ma"
            ema_alpha=0.2,     # if method="ema"
            ma_win=9           # if method="ma"
        )
        
        dash_first80_png = os.path.join(cfg.out_dir, "figs", f"dashboard_first80_loss_ba_epoch_{epoch:03d}.png")
        plot_dashboard_first80_loss_ba(
            history,
            dash_first80_png,
            title=f"GNN Adj First 80 Loss and BA @ epoch {epoch}",
            n_first=80
        )

        # qualitative trace (non-raster)
        save_one_val_trace_png(model, cfg, epoch, val_aug1_paths, device)

        # save all aug1 mats for MATLAB inspection
        save_val_aug1_matfiles(model, cfg, epoch, val_aug1_paths, device)

        # checkpoint
        save_checkpoint(cfg, model, opt, epoch, history)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} ba {train_ba:.3f} | "
            f"val(aug1) loss {val_loss:.4f} acc {val_acc:.3f} ba {val_ba:.3f} | "
            f"elapsed {dt/60:.1f} min"
        )

    print("Done.")
    return history


if __name__ == "__main__":
    cfg = make_cfg_from_cli()
    train(cfg)
