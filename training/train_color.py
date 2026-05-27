"""Train an EEGNet classifier on color-attention sessions.

Loads sessions recorded by `workspace/record_color.py`, slices each cue's
active window from the 8 raw channels, and trains an N-class classifier over
the cued color (e.g. RED/GREEN/BLUE/YELLOW). Saves the model state_dict and a
JSON sidecar describing the preprocessing so an inference script can
reproduce it exactly.

The default "active" window covers the full attention period — the all-LEDs
cue plus the per-color target display (default 0.5s + 1.5s = 2.0s) — because
`record_color.py` writes a single cue→reset gap that spans both phases.
Override with `--window_start_ms` / `--window_end_ms` if you want to train on
just one sub-phase (e.g. target-only at 500..2000).

Example calls (run from the repo root):

    # Single-session smoke test (4-class default).
    python training/train_color.py \\
        --name smoke_test \\
        --session_ids 2026-05-21_09-12-03 \\
        --epochs 5 --batch_size 4

    # Multi-session run.
    python training/train_color.py \\
        --name run_2026-05-21 \\
        --session_ids 2026-05-21_09-12-03 2026-05-21_09-30-11 \\
        --epochs 50 --batch_size 16 --lr 1e-3

    # Train only on the target window (skip the 0.5s cue phase).
    python training/train_color.py \\
        --name target_only \\
        --session_ids 2026-05-21_09-12-03 \\
        --window_start_ms 500 --window_end_ms 2000 \\
        --epochs 50
"""

import argparse
import json
import math
import os
import random
import sys
from typing import List, Tuple

# Allow `python training/train_color.py ...` (script mode) to resolve the
# `training` package by adding the repo root to sys.path. No-op when run as
# `python -m training.train_color`.
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
DEFAULT_COLORS = ["RED", "GREEN", "BLUE", "YELLOW"]
TIMING_TOLERANCE_MS = 50.0


def slice_windows(
    times: np.ndarray,
    samples: np.ndarray,
    events: pd.DataFrame,
    target_T: int,
    window_start_ms: float,
    window_end_ms: float,
    label_to_id: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for _, row in events.iterrows():
        if row["event"] != "cue":
            continue
        label = row.get("label")
        if label not in label_to_id:
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
        y_list.append(label_to_id[label])
    if not X_list:
        return (
            np.empty((0, len(CHANNELS), target_T), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(X_list), np.array(y_list, dtype=np.int64)


def build_dataset(
    session_ids: List[str],
    data_root: str,
    label_to_id: dict,
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
    Xs, ys = [], []
    for sid, times, samples, events in per_session:
        x, y = slice_windows(
            times, samples, events, target_T, window_start_ms, window_end_ms, label_to_id
        )
        print(f"  session {sid}: {x.shape[0]} episodes")
        Xs.append(x)
        ys.append(y)
    X = (
        np.concatenate(Xs, axis=0)
        if Xs
        else np.empty((0, len(CHANNELS), target_T), dtype=np.float32)
    )
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int64)
    return X, y, base


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[perm]
    return mixed, y, y[perm], lam


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al. 2018) generalised to N output classes."""

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 8,
        n_samples: int = 384,
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
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    label_to_id = {c: i for i, c in enumerate(args.colors)}
    print(f"Loading {len(args.session_ids)} session(s) from {args.data_root}/...")
    X, y, timing = build_dataset(
        args.session_ids,
        args.data_root,
        label_to_id,
        window_start_ms=args.window_start_ms,
        window_end_ms=args.window_end_ms,
    )
    if X.shape[0] == 0:
        raise RuntimeError("No training samples found.")
    effective_window_end_ms = (
        args.window_end_ms if args.window_end_ms is not None else timing.active_s * 1000.0
    )
    window_ms = effective_window_end_ms - args.window_start_ms
    counts = ", ".join(f"{c}={int((y == label_to_id[c]).sum())}" for c in args.colors)
    print(
        f"Total samples: {X.shape[0]} ({counts}). "
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
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx]) if len(val_idx) else None
    y_val = torch.from_numpy(y[val_idx]) if len(val_idx) else None

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(args.colors)
    model = EEGNet(n_classes=n_classes, n_channels=X.shape[1], n_samples=X.shape[2]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}, classes: {n_classes}, device: {device}")

    weight_decay = 1e-4
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    mixup_alpha = args.mixup_alpha

    history = {
        "epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }
    best_val_loss = float("inf")
    best_val_acc = float("nan")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if args.checkpoint_metric not in ("val_loss", "val_acc"):
        raise ValueError(
            f"--checkpoint_metric must be val_loss or val_acc (got {args.checkpoint_metric!r})"
        )

    def _is_better(curr_loss: float, curr_acc: float) -> bool:
        """Higher acc / lower loss wins; tiebreak on the *other* metric.

        Acc-based selection is preferred when val_loss climbs late in training
        from confidently-wrong predictions even as argmax accuracy keeps rising
        (overconfident-CE artifact).
        """
        if args.checkpoint_metric == "val_acc":
            if curr_acc > best_val_acc or math.isnan(best_val_acc):
                return True
            if curr_acc == best_val_acc and curr_loss < best_val_loss:
                return True
            return False
        # val_loss
        if curr_loss < best_val_loss:
            return True
        if curr_loss == best_val_loss and (
            math.isnan(best_val_acc) or curr_acc > best_val_acc
        ):
            return True
        return False

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            mixed_xb, ya, yb_perm, lam = mixup_batch(xb, yb, mixup_alpha)
            optim.zero_grad()
            logits = model(mixed_xb)
            loss = lam * loss_fn(logits, ya) + (1.0 - lam) * loss_fn(logits, yb_perm)
            loss.backward()
            optim.step()
            train_loss += loss.item() * xb.size(0)
            dominant = ya if lam >= 0.5 else yb_perm
            train_correct += (logits.argmax(1) == dominant).sum().item()
            train_total += xb.size(0)
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_loss = float("nan")
        val_acc = float("nan")
        if X_val is not None and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                logits = model(X_val.to(device))
                val_loss = loss_fn(logits, y_val.to(device)).item()
                val_acc = (logits.argmax(1).cpu() == y_val).float().mean().item()
            print(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f} acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
            )
            if _is_better(val_loss, val_acc):
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} acc={train_acc:.3f}")
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    model.load_state_dict(best_state)
    if X_val is not None and len(X_val) > 0:
        print(
            f"Best epoch {best_epoch} (selected by {args.checkpoint_metric}): "
            f"val_loss={best_val_loss:.4f} val_acc={best_val_acc:.3f}"
        )

    out_dir = os.path.join("training", "eegnet_models", args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "color_eegnet.pt")
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
        "label_to_id": label_to_id,
        "model": "EEGNet",
        "task": "color",
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
        "colors": args.colors,
        "max_trials": args.max_trials,
        "window_start_ms": float(args.window_start_ms),
        "window_end_ms": float(effective_window_end_ms),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": weight_decay,
        "mixup_alpha": mixup_alpha,
        "checkpoint_metric": args.checkpoint_metric,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "best_val_acc": float(best_val_acc) if not math.isnan(best_val_acc) else None,
        "seed": args.seed,
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
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
        y_train=y_train,
        X_val=X_val if has_val else None,
        y_val=y_val if has_val else None,
        labels=args.colors,
    )


def save_training_plots(
    out_dir: str,
    history: dict,
    model: nn.Module,
    device: torch.device,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val,
    y_val,
    labels: List[str],
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
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)

    chance = 1.0 / len(labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_acc"], label="train", color="tab:blue")
    if has_val:
        ax.plot(epochs, history["val_acc"], label="val", color="tab:orange")
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy.png"), dpi=120)
    plt.close(fig)

    model.eval()
    splits = [("train", X_train, y_train)]
    if has_val:
        splits.append(("val", X_val, y_val))
    n = len(labels)
    for split_name, X_split, y_split in splits:
        with torch.no_grad():
            logits = model(X_split.to(device))
            preds = logits.argmax(1).cpu().numpy()
        truth = y_split.numpy()
        cm = np.zeros((n, n), dtype=np.int64)
        for t, p in zip(truth, preds):
            cm[t, p] += 1

        # Row-normalised: each cell shows what fraction of the *true* class
        # landed in each predicted column. Diagonal == per-class recall.
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
        )
        overall_acc = (
            float(np.trace(cm)) / float(cm.sum()) if cm.sum() > 0 else float("nan")
        )

        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45 if n > 4 else 0, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(
            f"Confusion ({split_name}) — acc={overall_acc * 100:.1f}% (n={int(cm.sum())})"
        )
        for i in range(n):
            for j in range(n):
                txt_color = "white" if cm_pct[i, j] > 0.5 else "black"
                ax.text(
                    j, i, f"{int(cm[i, j])}\n{cm_pct[i, j] * 100:.1f}%",
                    ha="center", va="center", color=txt_color, fontsize=9,
                )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("row %")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"confusion_{split_name}.png"), dpi=120)
        plt.close(fig)

    print(f"Saved training plots -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train color-attention EEGNet classifier.")
    parser.add_argument(
        "--name", required=True,
        help="Training run name. Models are saved under training/eegnet_models/<name>/.",
    )
    parser.add_argument("--session_ids", nargs="+", required=True)
    parser.add_argument("--data_root", default="data/color")
    parser.add_argument(
        "--colors", nargs="+", default=DEFAULT_COLORS,
        help=f"Class labels and order (default: {' '.join(DEFAULT_COLORS)}).",
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
    parser.add_argument("--mixup_alpha", type=float, default=0.1)
    parser.add_argument(
        "--val_frac", type=float, default=0.1,
        help="Fraction of trials held out for validation (default 0.1 = 90/10).",
    )
    parser.add_argument(
        "--checkpoint_metric", choices=["val_loss", "val_acc"], default="val_acc",
        help="Which validation metric selects the saved checkpoint. "
             "Defaults to val_acc — overconfident-CE drift can make val_loss "
             "rise late even while argmax accuracy keeps climbing.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
