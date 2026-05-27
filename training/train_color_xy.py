"""Train an EEGNet regressor that predicts the target LED's (x, y) position.

Same color-attention sessions as `train_color.py`, but the target is the (x,y)
of the cued LED on the matrix (recorded as `cue_x` / `cue_y` in `events.csv`).
The model emits a 2-d continuous prediction in the matrix's pixel coordinate
frame; loss is MSE in normalized [0, 1] coords so the two axes contribute
comparably even when the matrix isn't square.

Saves the model state_dict and a JSON sidecar describing the preprocessing
and the layout dimensions used to normalize, so an inference script can
denormalize predictions back to pixel space.

Example calls (run from the repo root):

    # Default 32x16 matrix (matches workspace/record_color.py defaults).
    python training/train_color_xy.py \\
        --name smoke_test \\
        --session_ids 2026-05-21_09-12-03 \\
        --epochs 5 --batch_size 4

    # Multi-session run.
    python training/train_color_xy.py \\
        --name run_2026-05-21 \\
        --session_ids 2026-05-21_09-12-03 2026-05-21_09-30-11 \\
        --epochs 50 --batch_size 16 --lr 1e-3

    # Train only on the target window (skip the 0.5s cue phase).
    python training/train_color_xy.py \\
        --name target_only \\
        --session_ids 2026-05-21_09-12-03 \\
        --window_start_ms 500 --window_end_ms 2000
"""

import argparse
import json
import math
import os
import random
import sys
from typing import List, Tuple

# Allow `python training/train_color_xy.py ...` (script mode) to resolve the
# `training` package by adding the repo root to sys.path. No-op when run as
# `python -m training.train_color_xy`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training._session_io import (
    SessionTiming,
    check_cross_session_timing,
    load_session_csv,
    slice_cue_window,
)

CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
SAMPLING_RATE = 256
TIMING_TOLERANCE_MS = 50.0
DEFAULT_LAYOUT_W = 32
DEFAULT_LAYOUT_H = 16


def slice_windows(
    times: np.ndarray,
    samples: np.ndarray,
    events: pd.DataFrame,
    target_T: int,
    window_start_ms: float,
    window_end_ms: float,
    layout_w: int,
    layout_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut one (C, T) window per cue and return (X, Y) where Y is (n, 2) in [0,1].

    Skips cues whose `cue_x` / `cue_y` are missing or out of layout bounds.
    """
    X_list, Y_list = [], []
    for _, row in events.iterrows():
        if row["event"] != "cue":
            continue
        cx = row.get("cue_x")
        cy = row.get("cue_y")
        if pd.isna(cx) or pd.isna(cy):
            continue
        try:
            cx = int(cx)
            cy = int(cy)
        except (TypeError, ValueError):
            continue
        if not (0 <= cx < layout_w and 0 <= cy < layout_h):
            continue
        cue_ms = float(row["relative_time_ms"])
        win = slice_cue_window(
            times, samples, cue_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            target_T=target_T,
            pad=True,
        )
        if win is None:
            continue
        x = win.T  # (C, T)
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - mean) / std
        X_list.append(x.astype(np.float32))
        # Normalize to [0, 1] using the *count* of cells - 1 so corners hit 0/1.
        nx = cx / max(layout_w - 1, 1)
        ny = cy / max(layout_h - 1, 1)
        Y_list.append([nx, ny])
    if not X_list:
        return (
            np.empty((0, len(CHANNELS), target_T), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    return np.stack(X_list), np.asarray(Y_list, dtype=np.float32)


def build_dataset(
    session_ids: List[str],
    data_root: str,
    layout_w: int,
    layout_h: int,
    window_start_ms: float,
    window_end_ms: float,
) -> Tuple[np.ndarray, np.ndarray, SessionTiming]:
    timings: List[SessionTiming] = []
    per_session = []
    for sid in session_ids:
        session_dir = os.path.join(data_root, sid)
        times, samples, timing, events = load_session_csv(session_dir, CHANNELS)
        timings.append(timing)
        per_session.append((sid, times, samples, events))

    check_cross_session_timing(session_ids, timings, tolerance_ms=TIMING_TOLERANCE_MS)
    base = timings[0]
    active_ms = base.active_s * 1000.0

    if window_end_ms is None:
        window_end_ms = active_ms
    if window_start_ms < 0:
        raise ValueError(f"--window_start_ms must be >= 0 (got {window_start_ms})")
    if window_end_ms <= window_start_ms:
        raise ValueError(
            f"--window_end_ms ({window_end_ms}) must be > --window_start_ms ({window_start_ms})"
        )
    if window_end_ms > active_ms + TIMING_TOLERANCE_MS:
        raise ValueError(
            f"--window_end_ms ({window_end_ms}) exceeds active period "
            f"({active_ms:.1f} ms) for session {session_ids[0]}"
        )

    target_T = int(round((window_end_ms - window_start_ms) / 1000.0 * SAMPLING_RATE))
    Xs, Ys = [], []
    for sid, times, samples, events in per_session:
        x, yy = slice_windows(
            times, samples, events, target_T,
            window_start_ms, window_end_ms, layout_w, layout_h,
        )
        print(f"  session {sid}: {x.shape[0]} episodes")
        Xs.append(x)
        Ys.append(yy)
    X = (
        np.concatenate(Xs, axis=0)
        if Xs
        else np.empty((0, len(CHANNELS), target_T), dtype=np.float32)
    )
    Y = np.concatenate(Ys, axis=0) if Ys else np.empty((0, 2), dtype=np.float32)
    return X, Y, base


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    mixed_y = lam * y + (1.0 - lam) * y[perm]
    # Two-tap return mirrors the classify trainer; for regression we already
    # mixed the targets so the second/lam slot is unused but kept for shape.
    return mixed_x, mixed_y, mixed_y, 1.0


class EEGNetRegressor(nn.Module):
    """EEGNet trunk with a 2-d regression head (sigmoid -> [0, 1])."""

    def __init__(
        self,
        out_dim: int = 2,
        n_channels: int = 8,
        n_samples: int = 512,
        dropout: float = 0.25,
    ):
        super().__init__()
        F1, D, F2 = 8, 2, 16
        kernel_t = 64

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_t), padding=(0, kernel_t // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.separable(self.depthwise(self.block1(dummy)))
            flat = out.numel()
        self.head = nn.Linear(flat, out_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.head(x))


def _pixel_metrics(
    pred_norm: np.ndarray, target_norm: np.ndarray, layout_w: int, layout_h: int
) -> dict:
    """Mean / median Euclidean error after denormalizing back to pixel space."""
    pw = max(layout_w - 1, 1)
    ph = max(layout_h - 1, 1)
    px_pred = pred_norm * np.array([pw, ph])
    px_true = target_norm * np.array([pw, ph])
    err = np.linalg.norm(px_pred - px_true, axis=1)
    return {
        "mean_pixel_error": float(err.mean()) if err.size else float("nan"),
        "median_pixel_error": float(np.median(err)) if err.size else float("nan"),
        "max_pixel_error": float(err.max()) if err.size else float("nan"),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {len(args.session_ids)} session(s) from {args.data_root}/...")
    X, Y, timing = build_dataset(
        args.session_ids,
        args.data_root,
        layout_w=args.layout_w,
        layout_h=args.layout_h,
        window_start_ms=args.window_start_ms,
        window_end_ms=args.window_end_ms,
    )
    if X.shape[0] == 0:
        raise RuntimeError("No training samples found.")
    effective_window_end_ms = (
        args.window_end_ms if args.window_end_ms is not None else timing.active_s * 1000.0
    )
    window_ms = effective_window_end_ms - args.window_start_ms
    print(
        f"Total samples: {X.shape[0]} (layout {args.layout_w}x{args.layout_h}). "
        f"Window: {X.shape[2]} samples = {window_ms / 1000.0:.3f}s"
    )

    if not 0.0 <= args.val_frac < 1.0:
        raise ValueError(f"--val_frac must be in [0, 1) (got {args.val_frac})")
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    if args.max_trials is not None and args.max_trials < len(idx):
        idx = idx[: args.max_trials]
    cut = max(1, int(round((1.0 - args.val_frac) * len(idx))))
    train_idx, val_idx = idx[:cut], idx[cut:]

    X_train = torch.from_numpy(X[train_idx])
    Y_train = torch.from_numpy(Y[train_idx])
    X_val = torch.from_numpy(X[val_idx]) if len(val_idx) else None
    Y_val = torch.from_numpy(Y[val_idx]) if len(val_idx) else None

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetRegressor(
        out_dim=2, n_channels=X.shape[1], n_samples=X.shape[2]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}, device: {device}")

    weight_decay = 1e-4
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    mixup_alpha = args.mixup_alpha

    history = {
        "epoch": [], "train_loss": [], "train_pix_err": [],
        "val_loss": [], "val_pix_err": [],
    }
    best_val_loss = float("inf")
    best_val_pix = float("nan")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            mixed_xb, mixed_yb, _, _ = mixup_batch(xb, yb, mixup_alpha)
            optim.zero_grad()
            pred = model(mixed_xb)
            loss = loss_fn(pred, mixed_yb)
            loss.backward()
            optim.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_total += xb.size(0)
        train_loss = train_loss_sum / max(train_total, 1)

        model.eval()
        with torch.no_grad():
            train_pred_norm = model(X_train.to(device)).cpu().numpy()
        train_pix = _pixel_metrics(
            train_pred_norm, Y_train.numpy(), args.layout_w, args.layout_h
        )["mean_pixel_error"]

        val_loss = float("nan")
        val_pix = float("nan")
        if X_val is not None and len(X_val) > 0:
            with torch.no_grad():
                val_pred = model(X_val.to(device))
                val_loss = loss_fn(val_pred, Y_val.to(device)).item()
                val_pred_norm = val_pred.cpu().numpy()
            val_pix = _pixel_metrics(
                val_pred_norm, Y_val.numpy(), args.layout_w, args.layout_h
            )["mean_pixel_error"]
            print(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f} pix={train_pix:.2f}  "
                f"val_loss={val_loss:.4f} pix={val_pix:.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_pix = val_pix
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} pix={train_pix:.2f}")
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_pix_err"].append(train_pix)
        history["val_loss"].append(val_loss)
        history["val_pix_err"].append(val_pix)

    model.load_state_dict(best_state)
    if X_val is not None and len(X_val) > 0:
        print(
            f"Best epoch {best_epoch}: val_loss={best_val_loss:.4f} val_pix={best_val_pix:.2f}"
        )

    out_dir = os.path.join("training", "eegnet_models", args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "color_xy_eegnet.pt")
    torch.save(model.state_dict(), out_path)

    sidecar = {
        "channels": CHANNELS,
        "sampling_rate": SAMPLING_RATE,
        "active_s": timing.active_s,
        "reset_s": timing.reset_s,
        "window_start_ms": float(args.window_start_ms),
        "window_end_ms": float(effective_window_end_ms),
        "T": int(X.shape[2]),
        "normalization": "per_window_zscore",
        "model": "EEGNetRegressor",
        "task": "color_xy",
        "out_dim": 2,
        "out_activation": "sigmoid",
        "label_fields": ["cue_x", "cue_y"],
        "layout_w": int(args.layout_w),
        "layout_h": int(args.layout_h),
        "session_ids": args.session_ids,
    }
    sidecar_path = os.path.splitext(out_path)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Saved model -> {out_path}")
    print(f"Saved sidecar -> {sidecar_path}")

    train_params = {
        "name": args.name,
        "session_ids": args.session_ids,
        "data_root": args.data_root,
        "layout_w": int(args.layout_w),
        "layout_h": int(args.layout_h),
        "max_trials": args.max_trials,
        "window_start_ms": float(args.window_start_ms),
        "window_end_ms": float(effective_window_end_ms),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": weight_decay,
        "mixup_alpha": mixup_alpha,
        "checkpoint_strategy": "best_val_loss",
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "best_val_pix_err": float(best_val_pix) if not math.isnan(best_val_pix) else None,
        "seed": args.seed,
        "optimizer": "AdamW",
        "loss": "MSELoss(normalized_xy)",
        "val_frac": float(args.val_frac),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]) if X_val is not None else 0,
        "n_total": int(X.shape[0]),
        "device": str(device),
    }
    train_params_path = os.path.join(out_dir, "train_params.json")
    with open(train_params_path, "w") as f:
        json.dump(train_params, f, indent=2)
    print(f"Saved train params -> {train_params_path}")

    has_val = X_val is not None and len(X_val) > 0
    save_training_plots(
        out_dir=out_dir,
        history=history,
        model=model,
        device=device,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val if has_val else None,
        Y_val=Y_val if has_val else None,
        layout_w=args.layout_w,
        layout_h=args.layout_h,
    )


def save_training_plots(
    out_dir: str,
    history: dict,
    model: nn.Module,
    device: torch.device,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val,
    Y_val,
    layout_w: int,
    layout_h: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    has_val = X_val is not None and not all(math.isnan(v) for v in history["val_loss"])

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="train", color="tab:blue")
    if has_val:
        ax.plot(epochs, history["val_loss"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalized xy)")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_pix_err"], label="train", color="tab:blue")
    if has_val:
        ax.plot(epochs, history["val_pix_err"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean pixel error")
    ax.set_title("Pixel error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pixel_error.png"), dpi=120)
    plt.close(fig)

    model.eval()
    splits = [("train", X_train, Y_train)]
    if has_val:
        splits.append(("val", X_val, Y_val))
    for split_name, X_split, Y_split in splits:
        with torch.no_grad():
            pred = model(X_split.to(device)).cpu().numpy()
        truth = Y_split.numpy()
        pw = max(layout_w - 1, 1)
        ph = max(layout_h - 1, 1)
        pred_px = pred * np.array([pw, ph])
        true_px = truth * np.array([pw, ph])
        stats = _pixel_metrics(pred, truth, layout_w, layout_h)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        for (tx, ty), (px, py) in zip(true_px, pred_px):
            ax.plot([tx, px], [ty, py], color="gray", alpha=0.3, linewidth=0.6)
        ax.scatter(true_px[:, 0], true_px[:, 1], s=18, c="tab:blue", label="true")
        ax.scatter(pred_px[:, 0], pred_px[:, 1], s=18, c="tab:orange", label="pred")
        ax.set_xlim(-0.5, layout_w - 0.5)
        ax.set_ylim(layout_h - 0.5, -0.5)  # flip so (0,0) is top-left
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_title(
            f"{split_name}: mean={stats['mean_pixel_error']:.2f}px "
            f"median={stats['median_pixel_error']:.2f}px"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"scatter_{split_name}.png"), dpi=120)
        plt.close(fig)

        # Interactive HTML companion: click a pair to isolate it (highlight
        # the true/pred dots + connecting line, gray everything else).
        # No external deps — vanilla SVG + JS, opens in any browser.
        _save_interactive_scatter(
            os.path.join(out_dir, f"scatter_{split_name}.html"),
            split_name=split_name,
            true_px=true_px,
            pred_px=pred_px,
            layout_w=layout_w,
            layout_h=layout_h,
            stats=stats,
        )

    print(f"Saved training plots -> {out_dir}")


def _save_interactive_scatter(
    path: str,
    *,
    split_name: str,
    true_px: np.ndarray,
    pred_px: np.ndarray,
    layout_w: int,
    layout_h: int,
    stats: dict,
) -> None:
    """Write a self-contained HTML scatter that can isolate a single trial.

    Click any true/pred dot or the connecting line to highlight that trial
    pair; click empty space (or "show all") to reset. Hover any dot for
    per-trial pixel error.
    """
    import html
    import json as _json

    pairs = []
    for i, ((tx, ty), (px, py)) in enumerate(zip(true_px, pred_px)):
        err = float(np.hypot(px - tx, py - ty))
        pairs.append({
            "i": int(i),
            "tx": float(tx), "ty": float(ty),
            "px": float(px), "py": float(py),
            "err": err,
        })
    payload = {
        "pairs": pairs,
        "layout_w": int(layout_w),
        "layout_h": int(layout_h),
        "split": split_name,
        "mean_err": float(stats.get("mean_pixel_error", float("nan"))),
        "median_err": float(stats.get("median_pixel_error", float("nan"))),
        "max_err": float(stats.get("max_pixel_error", float("nan"))),
    }
    title = html.escape(
        f"{split_name}: mean={payload['mean_err']:.2f}px "
        f"median={payload['median_err']:.2f}px (n={len(pairs)})"
    )

    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>scatter __SPLIT__</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 13px/1.4 system-ui, sans-serif; margin: 16px; }
  .wrap { display: grid; grid-template-columns: minmax(420px, 1fr) 240px; gap: 16px; align-items: start; }
  .plot { border: 1px solid #ccc; background: #fff; }
  .panel { font-size: 13px; }
  .panel h3 { margin: 0 0 8px; font-size: 14px; }
  .panel p { margin: 4px 0; }
  .panel .muted { color: #666; }
  button { font: inherit; padding: 4px 10px; cursor: pointer; }
  .legend { display: flex; gap: 12px; align-items: center; margin: 6px 0 10px; flex-wrap: wrap; }
  .legend span.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
  .true { fill: #1f77b4; }
  .pred { fill: #ff7f0e; }
  .link { stroke: #888; stroke-width: 1; opacity: 0.35; }
  .grid { stroke: #eee; stroke-width: 1; }
  .frame { fill: none; stroke: #aaa; stroke-width: 1; }
  .dim { opacity: 0.07; }
  .hl { opacity: 1 !important; stroke-width: 2; }
  .hl-link { stroke: #d62728; stroke-width: 2; opacity: 1 !important; }
  .hover-ring { fill: none; stroke: #000; stroke-width: 1; pointer-events: none; }
</style></head>
<body>
<h2>__TITLE__</h2>
<div class="legend">
  <span><span class="dot" style="background:#1f77b4"></span>true</span>
  <span><span class="dot" style="background:#ff7f0e"></span>pred</span>
  <span class="muted">click any dot or connecting line to isolate; click empty space to reset</span>
</div>
<div class="wrap">
  <svg id="plot" class="plot" viewBox="0 0 540 320"
       preserveAspectRatio="xMidYMid meet" width="100%"></svg>
  <div class="panel">
    <h3>Selection</h3>
    <div id="info">No trial selected.</div>
    <p><button id="reset">Show all</button></p>
    <h3 style="margin-top:14px">Summary</h3>
    <p>n = <b id="n">0</b></p>
    <p>mean px err = <b id="mean">—</b></p>
    <p>median px err = <b id="median">—</b></p>
    <p>max px err = <b id="max">—</b></p>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
const W = DATA.layout_w, H = DATA.layout_h;

const svg = document.getElementById("plot");
const info = document.getElementById("info");

// Layout: leave room for axes and a small margin.
const PAD_L = 38, PAD_R = 12, PAD_T = 12, PAD_B = 28;
const VB_W = 540, VB_H = 320;
const PLOT_W = VB_W - PAD_L - PAD_R;
const PLOT_H = VB_H - PAD_T - PAD_B;

// Map pixel coords (0..W-1, 0..H-1) into SVG. Y is flipped so (0,0) is top-left.
function sx(x) { return PAD_L + (x / Math.max(W - 1, 1)) * PLOT_W; }
function sy(y) { return PAD_T + (y / Math.max(H - 1, 1)) * PLOT_H; }

// Light grid + frame (every cell, but coarsely thinned for big boards).
const stepX = W > 32 ? 4 : (W > 16 ? 2 : 1);
const stepY = H > 32 ? 4 : (H > 16 ? 2 : 1);
const ns = "http://www.w3.org/2000/svg";
function el(tag, attrs) {
  const e = document.createElementNS(ns, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
for (let x = 0; x < W; x += stepX) {
  svg.appendChild(el("line", {
    class: "grid",
    x1: sx(x), x2: sx(x), y1: PAD_T, y2: PAD_T + PLOT_H,
  }));
}
for (let y = 0; y < H; y += stepY) {
  svg.appendChild(el("line", {
    class: "grid",
    x1: PAD_L, x2: PAD_L + PLOT_W, y1: sy(y), y2: sy(y),
  }));
}
svg.appendChild(el("rect", {
  class: "frame",
  x: PAD_L, y: PAD_T, width: PLOT_W, height: PLOT_H,
}));
// Axis labels (corners only, to keep it readable).
function axis(text, x, y, anchor) {
  const t = el("text", { x, y, "font-size": 10, fill: "#666",
                         "text-anchor": anchor || "middle" });
  t.textContent = text;
  svg.appendChild(t);
}
axis("(0,0)", PAD_L, PAD_T - 2, "start");
axis(`(${W-1},0)`, PAD_L + PLOT_W, PAD_T - 2, "end");
axis(`(0,${H-1})`, PAD_L, PAD_T + PLOT_H + 14, "start");
axis(`(${W-1},${H-1})`, PAD_L + PLOT_W, PAD_T + PLOT_H + 14, "end");

// Data layers: links first (under dots), then dots.
const linkLayer = el("g", { id: "links" });
const dotLayer = el("g", { id: "dots" });
svg.appendChild(linkLayer);
svg.appendChild(dotLayer);

const linkEls = [];
const trueEls = [];
const predEls = [];

DATA.pairs.forEach((p, i) => {
  const ln = el("line", {
    class: "link", "data-i": i,
    x1: sx(p.tx), y1: sy(p.ty),
    x2: sx(p.px), y2: sy(p.py),
  });
  linkLayer.appendChild(ln);
  linkEls.push(ln);

  const ct = el("circle", {
    class: "true", "data-i": i, "data-kind": "true",
    cx: sx(p.tx), cy: sy(p.ty), r: 3.5,
  });
  const cp = el("circle", {
    class: "pred", "data-i": i, "data-kind": "pred",
    cx: sx(p.px), cy: sy(p.py), r: 3.5,
  });
  ct.appendChild(svgTitle(`#${i} true (${p.tx.toFixed(1)}, ${p.ty.toFixed(1)})  err=${p.err.toFixed(2)}px`));
  cp.appendChild(svgTitle(`#${i} pred (${p.px.toFixed(1)}, ${p.py.toFixed(1)})  err=${p.err.toFixed(2)}px`));
  dotLayer.appendChild(ct);
  dotLayer.appendChild(cp);
  trueEls.push(ct);
  predEls.push(cp);
});

function svgTitle(text) {
  const t = el("title");
  t.textContent = text;
  return t;
}

function isolate(i) {
  // Dim everything, un-dim the chosen pair, mark connector red.
  for (const arr of [linkEls, trueEls, predEls]) {
    arr.forEach((e, k) => {
      if (k === i) {
        e.classList.remove("dim");
        e.classList.add("hl");
        if (arr === linkEls) {
          e.classList.add("hl-link");
        }
      } else {
        e.classList.add("dim");
        e.classList.remove("hl");
        if (arr === linkEls) {
          e.classList.remove("hl-link");
        }
      }
    });
  }
  const p = DATA.pairs[i];
  info.innerHTML =
    `<p>trial <b>#${p.i}</b></p>` +
    `<p>true&nbsp;= (<b>${p.tx.toFixed(2)}</b>, <b>${p.ty.toFixed(2)}</b>)</p>` +
    `<p>pred&nbsp;= (<b>${p.px.toFixed(2)}</b>, <b>${p.py.toFixed(2)}</b>)</p>` +
    `<p>error = <b>${p.err.toFixed(2)} px</b></p>`;
}

function reset() {
  for (const arr of [linkEls, trueEls, predEls]) {
    arr.forEach(e => {
      e.classList.remove("dim");
      e.classList.remove("hl");
      e.classList.remove("hl-link");
    });
  }
  info.textContent = "No trial selected.";
}

document.getElementById("reset").addEventListener("click", reset);

// Click handling: any element with data-i isolates that trial; click on
// blank SVG resets.
svg.addEventListener("click", (ev) => {
  const t = ev.target;
  const idx = t && t.getAttribute && t.getAttribute("data-i");
  if (idx !== null && idx !== undefined) {
    isolate(parseInt(idx, 10));
  } else {
    reset();
  }
});

// Stats panel.
document.getElementById("n").textContent = DATA.pairs.length;
document.getElementById("mean").textContent =
  isFinite(DATA.mean_err) ? DATA.mean_err.toFixed(2) + " px" : "—";
document.getElementById("median").textContent =
  isFinite(DATA.median_err) ? DATA.median_err.toFixed(2) + " px" : "—";
document.getElementById("max").textContent =
  isFinite(DATA.max_err) ? DATA.max_err.toFixed(2) + " px" : "—";
</script>
</body></html>
"""
    rendered = (
        template
        .replace("__DATA_JSON__", _json.dumps(payload))
        .replace("__TITLE__", title)
        .replace("__SPLIT__", html.escape(split_name))
    )
    with open(path, "w") as f:
        f.write(rendered)


def main():
    parser = argparse.ArgumentParser(
        description="Train color-xy EEGNet regressor on (cue_x, cue_y)."
    )
    parser.add_argument(
        "--name", required=True,
        help="Training run name. Models are saved under training/eegnet_models/<name>/.",
    )
    parser.add_argument("--session_ids", nargs="+", required=True)
    parser.add_argument("--data_root", default="data/color")
    parser.add_argument(
        "--layout_w", type=int, default=DEFAULT_LAYOUT_W,
        help=f"Matrix width in pixels (default {DEFAULT_LAYOUT_W}).",
    )
    parser.add_argument(
        "--layout_h", type=int, default=DEFAULT_LAYOUT_H,
        help=f"Matrix height in pixels (default {DEFAULT_LAYOUT_H}).",
    )
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--window_start_ms", type=float, default=0.0)
    parser.add_argument(
        "--window_end_ms", type=float, default=None,
        help="End of per-cue window relative to cue onset. Default: full active period.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument(
        "--val_frac", type=float, default=0.1,
        help="Fraction of trials held out for validation (default 0.1 = 90/10).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
