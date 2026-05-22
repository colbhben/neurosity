"""Train a small EEGNet classifier on LEFT/RIGHT motor-imagery sessions.

Loads sessions recorded by `workspace/record_lr.py`, slices each cue's active
window from the 8 raw channels, and trains a binary classifier (0=LEFT,
1=RIGHT). Saves the model state_dict and a JSON sidecar describing the
preprocessing so `inference/inference_lr.py` can reproduce it exactly.

Example calls (run from the repo root):

    # Single-session smoke test.
    python training/train_lr.py \\
        --name smoke_test \\
        --session_ids 2026-05-19_17-14-52 \\
        --epochs 5 --batch_size 4

    # Multi-session training run.
    python training/train_lr.py \\
        --name run_2026-05-20 \\
        --session_ids 2026-05-19_17-14-52 2026-05-20_09-12-03 \\
        --epochs 50 --batch_size 16 --lr 1e-3

    # Override defaults explicitly.
    python training/train_lr.py \\
        --name baseline \\
        --session_ids 2026-05-19_17-14-52 \\
        --data_root data --seed 42 --epochs 30
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
SAMPLING_RATE = 256
LABEL_TO_ID = {"LEFT": 0, "RIGHT": 1}
TIMING_TOLERANCE_MS = 50.0


@dataclass
class SessionTiming:
    active_s: float
    reset_s: float


def infer_timing(events_df: pd.DataFrame) -> SessionTiming:
    """Infer active/reset durations from an events.csv DataFrame."""
    rows = events_df.to_dict("records")
    if len(rows) < 2:
        raise ValueError("events.csv has fewer than 2 rows; cannot infer timing")

    active_intervals = []
    reset_intervals = []
    for i, row in enumerate(rows):
        if row["event"] == "cue" and i + 1 < len(rows) and rows[i + 1]["event"] == "reset":
            active_intervals.append(rows[i + 1]["relative_time_ms"] - row["relative_time_ms"])
        if row["event"] == "reset" and i + 1 < len(rows) and rows[i + 1]["event"] == "cue":
            reset_intervals.append(rows[i + 1]["relative_time_ms"] - row["relative_time_ms"])

    if not active_intervals:
        raise ValueError("No cue->reset pairs found in events.csv")

    active_ms = float(np.mean(active_intervals))
    reset_ms = float(np.mean(reset_intervals)) if reset_intervals else 0.0
    return SessionTiming(active_s=active_ms / 1000.0, reset_s=reset_ms / 1000.0)


def load_session(
    session_dir: str,
) -> Tuple[np.ndarray, np.ndarray, SessionTiming, pd.DataFrame, pd.DataFrame]:
    data_path = os.path.join(session_dir, "data.csv")
    events_path = os.path.join(session_dir, "events.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    if not os.path.exists(events_path):
        raise FileNotFoundError(events_path)

    raw_cols = [f"raw_{c}" for c in CHANNELS]
    data = pd.read_csv(data_path, usecols=["relative_time_ms", *raw_cols])
    events = pd.read_csv(events_path)
    timing = infer_timing(events)

    times = data["relative_time_ms"].to_numpy()
    samples = data[raw_cols].to_numpy(dtype=np.float32)  # (N, 8)
    return times, samples, timing, events, data


def slice_windows(
    times: np.ndarray,
    samples: np.ndarray,
    events: pd.DataFrame,
    target_T: int,
    window_start_ms: float,
    window_end_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut one (8, target_T) window per cue. Returns (X, y).

    The slice covers cue_onset + [window_start_ms, window_end_ms).
    """
    X_list = []
    y_list = []
    for _, row in events.iterrows():
        if row["event"] != "cue":
            continue
        label = row["label"]
        if label not in LABEL_TO_ID:
            continue
        cue_ms = float(row["relative_time_ms"])
        t_start = cue_ms + window_start_ms
        t_end = cue_ms + window_end_ms
        mask = (times >= t_start) & (times < t_end)
        window = samples[mask]  # (T_actual, 8)
        if window.shape[0] == 0:
            continue
        if window.shape[0] >= target_T:
            window = window[:target_T]
        else:
            pad = np.zeros((target_T - window.shape[0], window.shape[1]), dtype=np.float32)
            window = np.concatenate([window, pad], axis=0)
        x = window.T  # (8, T)
        # Per-window per-channel z-score.
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - mean) / std
        X_list.append(x.astype(np.float32))
        y_list.append(LABEL_TO_ID[label])

    if not X_list:
        return np.empty((0, 8, target_T), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_list), np.array(y_list, dtype=np.int64)


def build_dataset(
    session_ids: List[str],
    data_root: str,
    window_start_ms: float = 0.0,
    window_end_ms: float = None,
) -> Tuple[np.ndarray, np.ndarray, SessionTiming]:
    timings: List[SessionTiming] = []
    per_session = []
    for sid in session_ids:
        session_dir = os.path.join(data_root, sid)
        times, samples, timing, events, _ = load_session(session_dir)
        timings.append(timing)
        per_session.append((sid, times, samples, events))

    # Cross-session timing consistency.
    base = timings[0]
    for sid, t in zip(session_ids, timings):
        if abs(t.active_s - base.active_s) * 1000 > TIMING_TOLERANCE_MS:
            raise ValueError(
                f"Session {sid} active_s={t.active_s:.3f} differs from "
                f"{session_ids[0]} active_s={base.active_s:.3f} "
                f"(tolerance {TIMING_TOLERANCE_MS}ms)"
            )
        if abs(t.reset_s - base.reset_s) * 1000 > TIMING_TOLERANCE_MS:
            raise ValueError(
                f"Session {sid} reset_s={t.reset_s:.3f} differs from "
                f"{session_ids[0]} reset_s={base.reset_s:.3f} "
                f"(tolerance {TIMING_TOLERANCE_MS}ms)"
            )

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
        x, y = slice_windows(times, samples, events, target_T, window_start_ms, window_end_ms)
        print(f"  session {sid}: {x.shape[0]} episodes")
        Xs.append(x)
        ys.append(y)
    X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, 8, target_T), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int64)
    return X, y, base


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Apply mixup (Zhang et al. 2018) to a batch.

    Returns (mixed_x, y_a, y_b, lam) so the caller can compute
    lam * loss(logits, y_a) + (1 - lam) * loss(logits, y_b).
    """
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[perm]
    return mixed, y, y[perm], lam


class EEGNet(nn.Module):
    """Compact CNN for binary EEG motor-imagery classification.

    Input shape: (B, 1, C=8, T). Output: (B, 2) logits.
    Roughly follows Lawhern et al. 2018 with F1=8, D=2, F2=16.
    """

    def __init__(self, n_channels: int = 8, n_samples: int = 768, dropout: float = 0.25):
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
        self.classifier = nn.Linear(flat, 2)

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
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

    print(f"Loading {len(args.session_ids)} session(s) from {args.data_root}/...")
    X, y, timing = build_dataset(
        args.session_ids,
        args.data_root,
        window_start_ms=args.window_start_ms,
        window_end_ms=args.window_end_ms,
    )
    if X.shape[0] == 0:
        raise RuntimeError("No training samples found.")
    window_ms = (
        args.window_end_ms if args.window_end_ms is not None else timing.active_s * 1000.0
    ) - args.window_start_ms
    print(
        f"Total samples: {X.shape[0]} "
        f"(LEFT={int((y == 0).sum())}, RIGHT={int((y == 1).sum())}). "
        f"Window: {X.shape[2]} samples = {window_ms / 1000.0:.3f}s"
    )

    # 80/20 episode-level split (operates on raw windows; head-specific
    # transforms are applied below). Shuffle is seeded so identical
    # `--max_trials` subsets across heads see identical train/val splits.
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    if args.max_trials is not None and args.max_trials < len(idx):
        idx = idx[: args.max_trials]
    cut = max(1, int(round(0.8 * len(idx))))
    train_idx, val_idx = idx[:cut], idx[cut:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx]) if len(val_idx) else None
    y_val = torch.from_numpy(y[val_idx]) if len(val_idx) else None

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(n_channels=X.shape[1], n_samples=X.shape[2]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}, device: {device}")

    # AdamW with weight decay regularizes the small EEGNet against the
    # ~20pt train/val gap observed on this dataset. Mixup further smooths
    # decision boundaries and is also active for the Riemann head.
    weight_decay = 1e-4
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    mixup_alpha = 0.1

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_val_acc = float("nan")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
            # Train accuracy reported against the dominant mixup label.
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
            if val_loss < best_val_loss:
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

    # Restore the best-val-loss weights before saving / plotting.
    model.load_state_dict(best_state)
    if X_val is not None and len(X_val) > 0:
        print(
            f"Best epoch {best_epoch}: val_loss={best_val_loss:.4f} val_acc={best_val_acc:.3f}"
        )

    out_dir = os.path.join("training", "models", args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lr_eegnet.pt")
    torch.save(model.state_dict(), out_path)
    effective_window_end_ms = (
        args.window_end_ms if args.window_end_ms is not None else timing.active_s * 1000.0
    )
    sidecar = {
        "channels": CHANNELS,
        "sampling_rate": SAMPLING_RATE,
        "active_s": timing.active_s,
        "reset_s": timing.reset_s,
        "window_start_ms": float(args.window_start_ms),
        "window_end_ms": float(effective_window_end_ms),
        "T": int(X.shape[2]),
        "normalization": "per_window_zscore",
        "label_to_id": LABEL_TO_ID,
        "model": "EEGNet",
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
        "best_val_acc": float(best_val_acc) if not math.isnan(best_val_acc) else None,
        "seed": args.seed,
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "train_val_split": 0.8,
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
) -> None:
    """Write loss/accuracy curves, confusion matrices, and a metrics CSV."""
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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_acc"], label="train", color="tab:blue")
    if has_val:
        ax.plot(epochs, history["val_acc"], label="val", color="tab:orange")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
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
    labels = sorted(LABEL_TO_ID, key=lambda k: LABEL_TO_ID[k])
    n = len(labels)
    for split_name, X_split, y_split in splits:
        with torch.no_grad():
            logits = model(X_split.to(device))
            preds = logits.argmax(1).cpu().numpy()
        truth = y_split.numpy()
        cm = np.zeros((n, n), dtype=np.int64)
        for t, p in zip(truth, preds):
            cm[t, p] += 1

        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix ({split_name})")
        thresh = cm.max() / 2 if cm.max() > 0 else 0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"confusion_{split_name}.png"), dpi=120)
        plt.close(fig)

    print(f"Saved training plots -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train LEFT/RIGHT EEGNet classifier.")
    parser.add_argument(
        "--name",
        required=True,
        help="Training run name. Models are saved under training/models/<name>/.",
    )
    parser.add_argument("--session_ids", nargs="+", required=True)
    parser.add_argument("--data_root", default="data/lr")
    parser.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help="If set, subsample (after seeded shuffle) to this many total "
        "trials before the 80/20 split. Used by sweep driver.",
    )
    parser.add_argument(
        "--window_start_ms",
        type=float,
        default=0.0,
        help="Start of the per-cue window relative to cue onset (ms).",
    )
    parser.add_argument(
        "--window_end_ms",
        type=float,
        default=None,
        help="End of the per-cue window relative to cue onset (ms). "
        "Defaults to the full active period.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
