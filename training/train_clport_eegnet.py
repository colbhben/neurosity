"""Train a two-head EEGNet that predicts (block_color, bowl_color) from
the entire goal-shown -> place-done EEG window of a HITL CLIPort session.

This is a fork of `training/train_lr.py`:
- Same EEGNet trunk (F1=8, D=2, F2=16) so weights can be reused as the
  EEG encoder in the EEG-conditioned policy.
- Two `nn.Linear(flat, 4)` heads predicting block & bowl colors over
  HITL_COLORS = ['blue','red','green','yellow'].
- Splits come from `training/clport/splits.make_splits` so val_seen,
  val_unseen, val_mixed are reported separately.

Example:
    python training/train_clport_eegnet.py \\
        --name clport_eegnet_v0 \\
        --session_dirs data/clport/2026-05-22_10-00-00 data/clport/2026-05-22_11-30-00 \\
        --epochs 50 --batch_size 16 --window_seconds 8.0
"""

import argparse
import json
import math
import os
import random
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running as `python training/train_clport_eegnet.py` from the repo
# root without having to `pip install -e .`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.clport.session_io import (  # noqa: E402
    CHANNELS, SAMPLING_RATE, build_xy, discover_sessions,
)
from training.clport.splits import (  # noqa: E402
    Episode, HITL_COLORS, color_to_idx, make_splits,
)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class EEGNetTrunk(nn.Module):
    """EEGNet feature extractor. Returns a flat (B, F) tensor.

    Identical to the trunk in training/train_lr.py:EEGNet so weights are
    interchangeable.
    """

    def __init__(self, n_channels: int = 8, n_samples: int = 2048,
                 dropout: float = 0.25):
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
            self.flat_dim = int(out.numel())

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return torch.flatten(x, 1)


class EEGNetTwoHead(nn.Module):
    """EEGNet trunk + two 4-class linear heads (block, bowl)."""

    def __init__(self, n_channels: int = 8, n_samples: int = 2048,
                 n_classes: int = 4, dropout: float = 0.25):
        super().__init__()
        self.trunk = EEGNetTrunk(n_channels, n_samples, dropout)
        self.block_head = nn.Linear(self.trunk.flat_dim, n_classes)
        self.bowl_head = nn.Linear(self.trunk.flat_dim, n_classes)

    def forward(self, x):
        feat = self.trunk(x)
        return self.block_head(feat), self.bowl_head(feat)


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------


def _evaluate(model, X, yb, yw, device):
    if X is None or len(X) == 0:
        return {"n": 0, "block_acc": float("nan"), "bowl_acc": float("nan"),
                "joint_acc": float("nan"), "block_loss": float("nan"),
                "bowl_loss": float("nan")}
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        block_logits, bowl_logits = model(X.to(device))
        block_loss = loss_fn(block_logits, yb.to(device)).item()
        bowl_loss = loss_fn(bowl_logits, yw.to(device)).item()
        block_pred = block_logits.argmax(1).cpu()
        bowl_pred = bowl_logits.argmax(1).cpu()
        block_acc = (block_pred == yb).float().mean().item()
        bowl_acc = (bowl_pred == yw).float().mean().item()
        joint_acc = ((block_pred == yb) & (bowl_pred == yw)).float().mean().item()
    return {
        "n": int(len(X)),
        "block_acc": block_acc, "bowl_acc": bowl_acc, "joint_acc": joint_acc,
        "block_loss": block_loss, "bowl_loss": bowl_loss,
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Discovering episodes across {len(args.session_dirs)} session(s)...")
    hitl_eps = discover_sessions(args.session_dirs)
    if args.success_only:
        n_before = len(hitl_eps)
        hitl_eps = [e for e in hitl_eps if e.success]
        print(f"  filtered to successful episodes: {n_before} -> {len(hitl_eps)}")
    if not hitl_eps:
        raise RuntimeError("No episodes discovered.")

    target_T = int(round(args.window_seconds * SAMPLING_RATE))
    print(f"  T = {target_T} samples ({args.window_seconds:.2f}s @ {SAMPLING_RATE}Hz)")

    split_eps = [
        Episode(
            session_id=e.session_id, episode_idx=e.episode_idx,
            block_color=e.block_color, bowl_color=e.bowl_color,
        ) for e in hitl_eps
    ]
    splits = make_splits(split_eps, seed=args.seed)
    # Map back to HitlEpisode by (session_id, episode_idx) for X-building.
    by_key = {(e.session_id, e.episode_idx): e for e in hitl_eps}
    split_hitl = {
        name: [by_key[(s.session_id, s.episode_idx)] for s in lst]
        for name, lst in splits.items()
    }
    for name, lst in split_hitl.items():
        print(f"  split {name}: {len(lst)} episodes")
    if not split_hitl["train"]:
        raise RuntimeError("Training split is empty; collect more sessions.")

    print("Building tensors...")
    X_train, yb_train, yw_train = build_xy(split_hitl["train"], target_T)
    eval_tensors = {}
    for name in ("val_seen", "val_unseen", "val_mixed"):
        Xe, yb, yw = build_xy(split_hitl[name], target_T)
        eval_tensors[name] = (
            torch.from_numpy(Xe) if len(Xe) else None,
            torch.from_numpy(yb) if len(Xe) else None,
            torch.from_numpy(yw) if len(Xe) else None,
        )

    X_train_t = torch.from_numpy(X_train)
    yb_train_t = torch.from_numpy(yb_train)
    yw_train_t = torch.from_numpy(yw_train)
    train_ds = TensorDataset(X_train_t, yb_train_t, yw_train_t)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetTwoHead(
        n_channels=len(CHANNELS),
        n_samples=target_T,
        n_classes=len(HITL_COLORS),
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())} | device: {device}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        n_seen = 0
        for xb, yb, yw in train_loader:
            xb, yb, yw = xb.to(device), yb.to(device), yw.to(device)
            optim.zero_grad()
            block_logits, bowl_logits = model(xb)
            loss = (
                args.block_weight * loss_fn(block_logits, yb)
                + args.bowl_weight * loss_fn(bowl_logits, yw)
            )
            loss.backward()
            optim.step()
            run_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = run_loss / max(n_seen, 1)

        ep_record = {"epoch": epoch, "train_loss": train_loss}
        for name, (Xe, yb, yw) in eval_tensors.items():
            metrics = _evaluate(model, Xe, yb, yw, device)
            for k, v in metrics.items():
                ep_record[f"{name}_{k}"] = v
        history.append(ep_record)

        seen_jacc = ep_record.get("val_seen_joint_acc", float("nan"))
        unseen_jacc = ep_record.get("val_unseen_joint_acc", float("nan"))
        print(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_seen joint={seen_jacc:.3f}  val_unseen joint={unseen_jacc:.3f}"
        )

        # Checkpoint on val_seen total loss (block + bowl) so we don't
        # over-rotate to either head.
        seen_total = (
            ep_record.get("val_seen_block_loss", float("inf"))
            + ep_record.get("val_seen_bowl_loss", float("inf"))
        )
        if seen_total < best_val_loss:
            best_val_loss = seen_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    model.load_state_dict(best_state)
    print(f"Best epoch (val_seen total loss): {best_epoch} ({best_val_loss:.4f})")

    out_dir = os.path.join("training", "eegnet_models", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # Save model + EEGNet-style sidecar so inference can reproduce
    # preprocessing exactly.
    out_path = os.path.join(out_dir, "clport_eegnet.pt")
    torch.save(model.state_dict(), out_path)
    sidecar = {
        "channels": CHANNELS,
        "sampling_rate": SAMPLING_RATE,
        "T": target_T,
        "window_seconds": args.window_seconds,
        "normalization": "per_window_zscore",
        "model": "EEGNetTwoHead",
        "n_classes": len(HITL_COLORS),
        "color_to_idx": color_to_idx,
        "hitl_colors": list(HITL_COLORS),
        "session_dirs": args.session_dirs,
        "success_only": args.success_only,
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    train_params = {
        "name": args.name,
        "session_dirs": args.session_dirs,
        "window_seconds": args.window_seconds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "block_weight": args.block_weight,
        "bowl_weight": args.bowl_weight,
        "best_epoch": best_epoch,
        "best_val_seen_total_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "seed": args.seed,
        "n_train": int(X_train.shape[0]),
        "n_val_seen": int(len(split_hitl["val_seen"])),
        "n_val_unseen": int(len(split_hitl["val_unseen"])),
        "n_val_mixed": int(len(split_hitl["val_mixed"])),
    }
    with open(os.path.join(out_dir, "train_params.json"), "w") as f:
        json.dump(train_params, f, indent=2)

    _save_plots(out_dir, history)
    print(f"Saved -> {out_dir}")


def _save_plots(out_dir: str, history: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    chance = 1.0 / len(HITL_COLORS)

    # Loss curves (train + per-split block+bowl loss).
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train", color="tab:blue")
    for name, color in [("val_seen", "tab:orange"), ("val_unseen", "tab:red"),
                        ("val_mixed", "tab:green")]:
        col_b = f"{name}_block_loss"
        col_w = f"{name}_bowl_loss"
        if col_b in df and col_w in df:
            ax.plot(df["epoch"], df[col_b] + df[col_w], label=name, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (block + bowl)")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)

    # Joint-accuracy curves per split.
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in [("val_seen", "tab:orange"), ("val_unseen", "tab:red"),
                        ("val_mixed", "tab:green")]:
        col = f"{name}_joint_acc"
        if col in df:
            ax.plot(df["epoch"], df[col], label=f"{name} joint", color=color)
    ax.axhline(chance ** 2, color="gray", linestyle="--",
               linewidth=1, label="joint chance")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Joint (block AND bowl) accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "joint_acc.png"), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Two-head HITL EEGNet trainer.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--session_dirs", nargs="+", required=True,
                        help="Paths to data/clport/<session> directories.")
    parser.add_argument("--window_seconds", type=float, default=8.0,
                        help="Goal-shown -> place-done EEG window length, "
                             "padded/truncated to this duration.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block_weight", type=float, default=1.0)
    parser.add_argument("--bowl_weight", type=float, default=1.0)
    parser.add_argument("--success_only", action="store_true",
                        help="Drop failed episodes before splitting.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
