"""Train an EEGNet classifier on item-mux motor-imagery sessions.

Loads sessions recorded by `workspace/record_item_mux.py`, slices each cue's
active window from the 8 raw channels, and trains a multi-task classifier
with two heads:
  - item: which of the N items was cued (e.g. PEN / HIGHLIGHTER).
  - location: which slot (0..N-1) the cued item occupied at cue time.

Saves the model state_dict and a JSON sidecar describing the preprocessing and
the canonical item ordering so `inference/inference_item_mux.py` can reproduce
it exactly.

Example calls (run from the repo root):

    # Single-session smoke test.
    python training/train_item_mux.py \\
        --name smoke_test \\
        --session_ids 2026-05-19_17-14-52 \\
        --epochs 5 --batch_size 4

    # Multi-session training run.
    python training/train_item_mux.py \\
        --name run_2026-05-20 \\
        --session_ids 2026-05-19_17-14-52 2026-05-20_09-12-03 \\
        --epochs 50 --batch_size 16 --lr 1e-3
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
TIMING_TOLERANCE_MS = 50.0


@dataclass
class SessionTiming:
    active_s: float
    reset_s: float


def infer_timing(events_df: pd.DataFrame) -> SessionTiming:
    """Infer active/reset durations from an events.csv DataFrame.

    Uses cue->reset gaps for active_s and reset->cue gaps for reset_s.
    """
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


def session_items(events: pd.DataFrame) -> List[str]:
    """Return canonical sorted item list for a session (from any reset row)."""
    resets = events[events["event"] == "reset"]
    layouts = resets["locations"].dropna().astype(str)
    layouts = layouts[layouts.str.len() > 0]
    if layouts.empty:
        raise ValueError("events.csv has no reset rows with a 'locations' field")
    items = sorted(set(layouts.iloc[0].split(";")))
    for layout in layouts:
        if sorted(set(layout.split(";"))) != items:
            raise ValueError("inconsistent item set across reset rows")
    return items


def load_session(
    session_dir: str,
) -> Tuple[np.ndarray, np.ndarray, SessionTiming, pd.DataFrame, pd.DataFrame, List[str]]:
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
    items = session_items(events)

    times = data["relative_time_ms"].to_numpy()
    samples = data[raw_cols].to_numpy(dtype=np.float32)  # (N, 8)
    return times, samples, timing, events, data, items


def slice_windows(
    times: np.ndarray,
    samples: np.ndarray,
    events: pd.DataFrame,
    target_T: int,
    active_s: float,
    item_to_id: dict,
    n_locations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut one (8, target_T) window per cue. Returns (X, y_item, y_loc)."""
    X_list = []
    y_item: List[int] = []
    y_loc: List[int] = []
    active_ms = active_s * 1000.0
    for _, row in events.iterrows():
        if row["event"] != "cue":
            continue
        label = row["label"]
        if label not in item_to_id:
            continue
        loc = int(row["location"])
        if loc < 0 or loc >= n_locations:
            continue
        t_start = float(row["relative_time_ms"])
        t_end = t_start + active_ms
        mask = (times >= t_start) & (times < t_end)
        window = samples[mask]
        if window.shape[0] == 0:
            continue
        if window.shape[0] >= target_T:
            window = window[:target_T]
        else:
            pad = np.zeros((target_T - window.shape[0], window.shape[1]), dtype=np.float32)
            window = np.concatenate([window, pad], axis=0)
        x = window.T  # (8, T)
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - mean) / std
        X_list.append(x.astype(np.float32))
        y_item.append(item_to_id[label])
        y_loc.append(loc)

    if not X_list:
        return (
            np.empty((0, 8, target_T), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.stack(X_list),
        np.array(y_item, dtype=np.int64),
        np.array(y_loc, dtype=np.int64),
    )


def build_dataset(
    session_ids: List[str], data_root: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SessionTiming, List[str]]:
    timings: List[SessionTiming] = []
    per_session = []
    items_canonical: List[str] = None
    for sid in session_ids:
        session_dir = os.path.join(data_root, sid)
        times, samples, timing, events, _, items = load_session(session_dir)
        if items_canonical is None:
            items_canonical = items
        elif items != items_canonical:
            raise ValueError(
                f"Session {sid} item set {items} differs from "
                f"{session_ids[0]} item set {items_canonical}"
            )
        timings.append(timing)
        per_session.append((sid, times, samples, events))

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

    item_to_id = {name: i for i, name in enumerate(items_canonical)}
    n_locations = len(items_canonical)

    target_T = int(round(base.active_s * SAMPLING_RATE))
    Xs, yi, yl = [], [], []
    for sid, times, samples, events in per_session:
        x, ya, yb = slice_windows(
            times, samples, events, target_T, base.active_s, item_to_id, n_locations
        )
        print(f"  session {sid}: {x.shape[0]} episodes")
        Xs.append(x)
        yi.append(ya)
        yl.append(yb)
    X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, 8, target_T), dtype=np.float32)
    y_item = np.concatenate(yi, axis=0) if yi else np.empty((0,), dtype=np.int64)
    y_loc = np.concatenate(yl, axis=0) if yl else np.empty((0,), dtype=np.int64)
    return X, y_item, y_loc, base, items_canonical


class EEGNetMux(nn.Module):
    """Compact EEGNet backbone with two classification heads.

    Input shape: (B, 1, C=8, T). Outputs: item logits (B, n_items) and location
    logits (B, n_locations). Backbone roughly follows Lawhern et al. 2018 with
    F1=8, D=2, F2=16.
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_samples: int = 768,
        n_items: int = 2,
        n_locations: int = 2,
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
        self.item_head = nn.Linear(flat, n_items)
        self.loc_head = nn.Linear(flat, n_locations)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = torch.flatten(x, 1)
        return self.item_head(x), self.loc_head(x)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {len(args.session_ids)} session(s) from {args.data_root}/...")
    X, y_item, y_loc, timing, items = build_dataset(args.session_ids, args.data_root)
    if X.shape[0] == 0:
        raise RuntimeError("No training samples found.")
    n_items = len(items)
    n_locations = n_items
    item_counts = {items[i]: int((y_item == i).sum()) for i in range(n_items)}
    loc_counts = {i: int((y_loc == i).sum()) for i in range(n_locations)}
    print(
        f"Total samples: {X.shape[0]}. Items={items}. "
        f"Item counts={item_counts}. Location counts={loc_counts}. "
        f"Window: {X.shape[2]} samples = {timing.active_s:.3f}s"
    )

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    cut = max(1, int(round(0.8 * len(idx))))
    train_idx, val_idx = idx[:cut], idx[cut:]
    X_train = torch.from_numpy(X[train_idx])
    yi_train = torch.from_numpy(y_item[train_idx])
    yl_train = torch.from_numpy(y_loc[train_idx])
    X_val = torch.from_numpy(X[val_idx]) if len(val_idx) else None
    yi_val = torch.from_numpy(y_item[val_idx]) if len(val_idx) else None
    yl_val = torch.from_numpy(y_loc[val_idx]) if len(val_idx) else None

    train_ds = TensorDataset(X_train, yi_train, yl_train)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetMux(
        n_channels=X.shape[1],
        n_samples=X.shape[2],
        n_items=n_items,
        n_locations=n_locations,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}, device: {device}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "epoch": [],
        "train_loss": [], "train_loss_item": [], "train_loss_loc": [],
        "train_acc_item": [], "train_acc_loc": [],
        "val_loss": [], "val_loss_item": [], "val_loss_loc": [],
        "val_acc_item": [], "val_acc_loc": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "loss_i": 0.0, "loss_l": 0.0,
                "correct_i": 0, "correct_l": 0, "total": 0}
        for xb, yib, ylb in train_loader:
            xb = xb.to(device)
            yib = yib.to(device)
            ylb = ylb.to(device)
            optim.zero_grad()
            logits_i, logits_l = model(xb)
            loss_i = loss_fn(logits_i, yib)
            loss_l = loss_fn(logits_l, ylb)
            loss = loss_i + loss_l
            loss.backward()
            optim.step()
            bs = xb.size(0)
            sums["loss"] += loss.item() * bs
            sums["loss_i"] += loss_i.item() * bs
            sums["loss_l"] += loss_l.item() * bs
            sums["correct_i"] += (logits_i.argmax(1) == yib).sum().item()
            sums["correct_l"] += (logits_l.argmax(1) == ylb).sum().item()
            sums["total"] += bs
        n = max(sums["total"], 1)
        train_loss = sums["loss"] / n
        train_loss_i = sums["loss_i"] / n
        train_loss_l = sums["loss_l"] / n
        train_acc_i = sums["correct_i"] / n
        train_acc_l = sums["correct_l"] / n

        val_loss = float("nan")
        val_loss_i = float("nan")
        val_loss_l = float("nan")
        val_acc_i = float("nan")
        val_acc_l = float("nan")
        if X_val is not None and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                logits_i, logits_l = model(X_val.to(device))
                val_loss_i = loss_fn(logits_i, yi_val.to(device)).item()
                val_loss_l = loss_fn(logits_l, yl_val.to(device)).item()
                val_loss = val_loss_i + val_loss_l
                val_acc_i = (logits_i.argmax(1).cpu() == yi_val).float().mean().item()
                val_acc_l = (logits_l.argmax(1).cpu() == yl_val).float().mean().item()
            print(
                f"epoch {epoch:3d}  "
                f"train: loss={train_loss:.4f} item_acc={train_acc_i:.3f} loc_acc={train_acc_l:.3f}  "
                f"val: loss={val_loss:.4f} item_acc={val_acc_i:.3f} loc_acc={val_acc_l:.3f}"
            )
        else:
            print(
                f"epoch {epoch:3d}  train: loss={train_loss:.4f} "
                f"item_acc={train_acc_i:.3f} loc_acc={train_acc_l:.3f}"
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_loss_item"].append(train_loss_i)
        history["train_loss_loc"].append(train_loss_l)
        history["train_acc_item"].append(train_acc_i)
        history["train_acc_loc"].append(train_acc_l)
        history["val_loss"].append(val_loss)
        history["val_loss_item"].append(val_loss_i)
        history["val_loss_loc"].append(val_loss_l)
        history["val_acc_item"].append(val_acc_i)
        history["val_acc_loc"].append(val_acc_l)

    out_dir = os.path.join("training", "models", args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "item_mux_eegnet.pt")
    torch.save(model.state_dict(), out_path)
    sidecar = {
        "channels": CHANNELS,
        "sampling_rate": SAMPLING_RATE,
        "active_s": timing.active_s,
        "reset_s": timing.reset_s,
        "T": int(X.shape[2]),
        "normalization": "per_window_zscore",
        "items": items,
        "n_locations": n_locations,
        "model": "EEGNetMux",
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss(item) + CrossEntropyLoss(loc)",
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
        items=items,
        n_locations=n_locations,
        X_train=X_train,
        yi_train=yi_train,
        yl_train=yl_train,
        X_val=X_val if has_val else None,
        yi_val=yi_val if has_val else None,
        yl_val=yl_val if has_val else None,
    )


def save_training_plots(
    out_dir: str,
    history: dict,
    model: nn.Module,
    device: torch.device,
    items: List[str],
    n_locations: int,
    X_train: torch.Tensor,
    yi_train: torch.Tensor,
    yl_train: torch.Tensor,
    X_val,
    yi_val,
    yl_val,
) -> None:
    """Write loss/accuracy curves, per-head confusion matrices, and metrics CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    has_val = X_val is not None and not all(math.isnan(v) for v in history["val_loss"])

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # Combined loss.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="train (item+loc)", color="tab:blue")
    if has_val:
        ax.plot(epochs, history["val_loss"], label="val (item+loc)", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss (sum of heads)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)

    # Per-head loss curves.
    for head_name, train_key, val_key in [
        ("item", "train_loss_item", "val_loss_item"),
        ("loc", "train_loss_loc", "val_loss_loc"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history[train_key], label="train", color="tab:blue")
        if has_val:
            ax.plot(epochs, history[val_key], label="val", color="tab:orange")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss")
        ax.set_title(f"Loss ({head_name} head)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"loss_{head_name}.png"), dpi=120)
        plt.close(fig)

    # Per-head accuracy.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_acc_item"], label="train item", color="tab:blue")
    ax.plot(epochs, history["train_acc_loc"], label="train loc", color="tab:cyan")
    if has_val:
        ax.plot(epochs, history["val_acc_item"], label="val item", color="tab:orange")
        ax.plot(epochs, history["val_acc_loc"], label="val loc", color="tab:red")
    ax.axhline(1.0 / max(len(items), 1), color="gray", linestyle="--", linewidth=1,
               label=f"item chance ({1.0 / max(len(items), 1):.2f})")
    if n_locations != len(items):
        ax.axhline(1.0 / max(n_locations, 1), color="gray", linestyle=":", linewidth=1,
                   label=f"loc chance ({1.0 / max(n_locations, 1):.2f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Accuracy (per head)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy.png"), dpi=120)
    plt.close(fig)

    model.eval()
    splits = [("train", X_train, yi_train, yl_train)]
    if has_val:
        splits.append(("val", X_val, yi_val, yl_val))

    item_labels = items
    loc_labels = [str(i) for i in range(n_locations)]

    for split_name, X_split, yi_split, yl_split in splits:
        with torch.no_grad():
            li, ll = model(X_split.to(device))
            preds_i = li.argmax(1).cpu().numpy()
            preds_l = ll.argmax(1).cpu().numpy()
        truth_i = yi_split.numpy()
        truth_l = yl_split.numpy()

        for head_name, labels, truth, preds in [
            ("item", item_labels, truth_i, preds_i),
            ("loc", loc_labels, truth_l, preds_l),
        ]:
            n = len(labels)
            cm = np.zeros((n, n), dtype=np.int64)
            for t, p in zip(truth, preds):
                cm[t, p] += 1
            total = int(cm.sum())
            acc = (np.trace(cm) / total) if total else 0.0

            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(
                f"Confusion matrix ({head_name}, {split_name}, acc={acc:.3f})"
            )
            # Use the colormap's actual normalization range so light cells get
            # dark text even when they're nonzero (e.g. cm in [14, 25]).
            cmin = float(cm.min())
            cmax = float(cm.max())
            for i in range(n):
                for j in range(n):
                    norm = (cm[i, j] - cmin) / (cmax - cmin) if cmax > cmin else 0.0
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if norm > 0.6 else "black",
                        clip_on=False,
                    )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"confusion_{head_name}_{split_name}.png"),
                dpi=120,
            )
            plt.close(fig)

    print(f"Saved training plots -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train item-mux EEGNet classifier.")
    parser.add_argument(
        "--name",
        required=True,
        help="Training run name. Models are saved under training/models/<name>/.",
    )
    parser.add_argument("--session_ids", nargs="+", required=True)
    parser.add_argument("--data_root", default="data/item_mux")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
