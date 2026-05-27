"""Train DANCE on color-attention sessions.

Same task as `training/train_color.py` (4-way classify over the cued color)
but using Meta FAIR's DANCE architecture (`third_party/dance`) with
`use_channel_merger=False` so DANCE's spatial Fourier projection is bypassed
and our 8-channel Crown layout feeds the conv stack directly.

Each color trial is a single full-window event (`start=0`, `end=1`,
`class∈{1..N}`) — DANCE's degenerate "1 event per window" case. Background
class 0 is reserved for padding queries; total class count handed to DANCE is
`len(--colors) + 1`.

Setup (run once from repo root):

    pip install -e third_party/dance

Example calls:

    # Single-session smoke test (4-class default).
    python training/train_color_dance.py \\
        --name dance_smoke \\
        --session_ids 2026-05-21_09-12-03 \\
        --epochs 5 --batch_size 4

    # Multi-session run.
    python training/train_color_dance.py \\
        --name run_2026-05-27 \\
        --session_ids 2026-05-27_11-46-59 2026-05-27_11-56-54 \\
                      2026-05-27_12-55-39 2026-05-27_13-04-17 \\
        --epochs 50 --batch_size 16 --lr 5e-5
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List

# Allow `python training/train_color_dance.py ...` (script mode) to resolve
# the `training` package by adding the repo root to sys.path. No-op when run
# as `python -m training.train_color_dance`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from training.dance_io.dataset import (
    CHANNELS,
    SAMPLING_RATE,
    ColorDanceDataset,
    build_color_trials,
    make_collate,
)

DEFAULT_COLORS = ["RED", "GREEN", "BLUE", "YELLOW"]


def _import_dance():
    """Import dance with a friendly error if the submodule isn't installed."""
    try:
        from dance import Dance  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import `dance`. Install the submodule with:\n"
            "    pip install -e third_party/dance\n"
            f"(underlying error: {exc})"
        ) from exc
    return Dance


def _move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _trial_logits(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Reduce DANCE per-token dense logits to a single per-trial logit vector.

    Each trial is a single full-window event, so averaging the dense head
    logits across the temporal axis is a robust, postproc-free way to get
    trial-level class scores. Returns (B, n_classes).
    """
    return out["pred_dense"].mean(dim=1)


def _accuracy_and_loss(
    model,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run the model over `loader`, return mean loss + non-bg trial accuracy."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            out = model(batch)
            total_loss += float(out["loss"].item())
            n_batches += 1
            logits = _trial_logits(out)  # (B, n_classes), index 0 = background
            # Compare against the slot-0 ground-truth class (the one full-window event).
            preds = logits[:, 1:].argmax(dim=-1) + 1
            truth = batch["class"][:, 0]
            correct += int((preds == truth).sum().item())
            total += int(truth.numel())
    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": correct / max(total, 1),
    }


def _is_better(
    metric: str,
    curr_loss: float,
    curr_acc: float,
    best_loss: float,
    best_acc: float,
) -> bool:
    """Match `train_color.py`'s checkpoint policy: higher acc / lower loss
    wins, with the *other* metric as tiebreaker.
    """
    if metric == "val_acc":
        if math.isnan(best_acc) or curr_acc > best_acc:
            return True
        if curr_acc == best_acc and curr_loss < best_loss:
            return True
        return False
    # val_loss
    if curr_loss < best_loss:
        return True
    if curr_loss == best_loss and (math.isnan(best_acc) or curr_acc > best_acc):
        return True
    return False


def train(args):
    Dance = _import_dance()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    label_to_id = {c: i for i, c in enumerate(args.colors)}
    print(f"Loading {len(args.session_ids)} session(s) from {args.data_root}/...")
    trials, timing, window_seconds = build_color_trials(
        args.session_ids,
        args.data_root,
        label_to_id,
        window_start_ms=args.window_start_ms,
        window_end_ms=args.window_end_ms,
    )
    if not trials:
        raise RuntimeError("No training samples found.")

    # Normalize for printing + sidecar.
    effective_window_end_ms = (
        args.window_end_ms if args.window_end_ms is not None else timing.active_s * 1000.0
    )
    target_T = int(round(window_seconds * SAMPLING_RATE))
    counts = ", ".join(
        f"{c}={sum(1 for t in trials if t.class_id == label_to_id[c] + 1)}"
        for c in args.colors
    )
    print(
        f"Total samples: {len(trials)} ({counts}). "
        f"Window: {target_T} samples = {window_seconds:.3f}s"
    )

    # Episode-level shuffle / split (seeded so different runs can be compared).
    if not 0.0 <= args.val_frac < 1.0:
        raise ValueError(f"--val_frac must be in [0, 1) (got {args.val_frac})")
    idx = np.arange(len(trials))
    np.random.shuffle(idx)
    if args.max_trials is not None and args.max_trials < len(idx):
        idx = idx[: args.max_trials]
    cut = max(1, int(round((1.0 - args.val_frac) * len(idx))))
    train_idx, val_idx = idx[:cut].tolist(), idx[cut:].tolist()

    full_ds = ColorDanceDataset(trials)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx) if val_idx else None

    n_queries = args.n_queries
    collate = make_collate(n_queries)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, collate_fn=collate,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            drop_last=False, collate_fn=collate,
        )
        if val_ds is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(args.colors) + 1  # +1 for background/padding (class 0)
    model = Dance(
        n_channels=len(CHANNELS),
        n_classes=n_classes,
        n_queries=n_queries,
        duration=window_seconds,
        use_channel_merger=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model params: {n_params}, classes: {n_classes} "
        f"(incl. background), n_queries: {n_queries}, device: {device}"
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "train_detr_class": [], "train_detr_iou": [],
        "train_dense": [], "train_consistency": [],
    }
    best_val_loss = float("inf")
    best_val_acc = float("nan")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if args.checkpoint_metric not in ("val_loss", "val_acc"):
        raise ValueError(
            f"--checkpoint_metric must be val_loss or val_acc "
            f"(got {args.checkpoint_metric!r})"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {
            "loss": 0.0,
            "detr_class": 0.0, "detr_iou": 0.0,
            "dense": 0.0, "consistency": 0.0,
        }
        n_batches = 0
        correct = 0
        total = 0
        for batch in train_loader:
            batch = _move(batch, device)
            optim.zero_grad()
            out = model(batch)
            out["loss"].backward()
            optim.step()
            running["loss"] += float(out["loss"].item())
            details = out.get("loss_details", {})
            for k in ("detr_class", "detr_iou", "dense", "consistency"):
                v = details.get(k)
                if v is not None:
                    running[k] += float(v.item() if hasattr(v, "item") else v)
            n_batches += 1
            logits = _trial_logits(out)
            preds = logits[:, 1:].argmax(dim=-1) + 1
            truth = batch["class"][:, 0]
            correct += int((preds == truth).sum().item())
            total += int(truth.numel())

        train_loss = running["loss"] / max(n_batches, 1)
        train_acc = correct / max(total, 1)
        for k in ("detr_class", "detr_iou", "dense", "consistency"):
            running[k] /= max(n_batches, 1)

        val_loss = float("nan")
        val_acc = float("nan")
        if val_loader is not None:
            stats = _accuracy_and_loss(model, val_loader, device)
            val_loss, val_acc = stats["loss"], stats["acc"]
            print(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f} acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
            )
            if _is_better(
                args.checkpoint_metric, val_loss, val_acc, best_val_loss, best_val_acc
            ):
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
        history["train_detr_class"].append(running["detr_class"])
        history["train_detr_iou"].append(running["detr_iou"])
        history["train_dense"].append(running["dense"])
        history["train_consistency"].append(running["consistency"])

    model.load_state_dict(best_state)
    if val_loader is not None:
        print(
            f"Best epoch {best_epoch} (selected by {args.checkpoint_metric}): "
            f"val_loss={best_val_loss:.4f} val_acc={best_val_acc:.3f}"
        )

    out_dir = os.path.join("training", "dance_models", args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "color_dance.pt")
    torch.save(model.state_dict(), out_path)

    sidecar = {
        "channels": CHANNELS,
        "sampling_rate": SAMPLING_RATE,
        "active_s": timing.active_s,
        "reset_s": timing.reset_s,
        "window_start_ms": float(args.window_start_ms),
        "window_end_ms": float(effective_window_end_ms),
        "window_seconds": float(window_seconds),
        "T": int(target_T),
        "normalization": "per_window_zscore",
        "label_to_id": label_to_id,
        "n_classes_with_background": n_classes,
        "n_queries": int(n_queries),
        "use_channel_merger": False,
        "model": "Dance",
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
        "window_seconds": float(window_seconds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "n_queries": int(n_queries),
        "use_channel_merger": False,
        "checkpoint_metric": args.checkpoint_metric,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "best_val_acc": float(best_val_acc) if not math.isnan(best_val_acc) else None,
        "seed": args.seed,
        "optimizer": "AdamW",
        "loss": "Dance(DETR + dense + consistency)",
        "val_frac": float(args.val_frac),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_total": len(trials),
        "device": str(device),
    }
    train_params_path = os.path.join(out_dir, "train_params.json")
    with open(train_params_path, "w") as f:
        json.dump(train_params, f, indent=2)
    print(f"Saved train params -> {train_params_path}")

    save_training_plots(
        out_dir=out_dir,
        history=history,
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        labels=args.colors,
    )


def save_training_plots(
    out_dir: str,
    history: dict,
    model,
    device: torch.device,
    train_loader: DataLoader,
    val_loader,
    labels: List[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    has_val = val_loader is not None and not all(math.isnan(v) for v in history["val_loss"])

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # Total loss.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="train", color="tab:blue")
    if has_val:
        ax.plot(epochs, history["val_loss"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("DANCE total loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)

    # Per-component DANCE losses (training only — components aren't returned
    # in eval mode).
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, color in [
        ("train_detr_class", "tab:blue"),
        ("train_detr_iou", "tab:orange"),
        ("train_dense", "tab:green"),
        ("train_consistency", "tab:red"),
    ]:
        ax.plot(epochs, history[k], label=k.replace("train_", ""), color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss component")
    ax.set_title("Per-component training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_components.png"), dpi=120)
    plt.close(fig)

    # Accuracy.
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

    # Confusion matrices (train + val if present), same format as
    # train_color.py: count + row-% per cell, overall acc in title.
    splits = [("train", train_loader)]
    if has_val:
        splits.append(("val", val_loader))
    n = len(labels)
    model.eval()
    for split_name, loader in splits:
        truths: list = []
        preds_all: list = []
        with torch.no_grad():
            for batch in loader:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                out = model(batch_dev)
                logits = _trial_logits(out)  # (B, n_classes)
                preds = logits[:, 1:].argmax(dim=-1).cpu().numpy()  # 0..N-1
                truth = batch["class"][:, 0].cpu().numpy() - 1  # back to 0..N-1
                truths.append(truth)
                preds_all.append(preds)
        if not truths:
            continue
        truth = np.concatenate(truths)
        pred = np.concatenate(preds_all)
        cm = np.zeros((n, n), dtype=np.int64)
        for t, p in zip(truth, pred):
            if 0 <= t < n and 0 <= p < n:
                cm[t, p] += 1

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
    parser = argparse.ArgumentParser(
        description="Train DANCE classifier on color-attention sessions."
    )
    parser.add_argument(
        "--name", required=True,
        help="Training run name. Models are saved under training/dance_models/<name>/.",
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="AdamW learning rate (DANCE example default 5e-5).",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3,
        help="AdamW weight decay (DANCE example default 1e-3).",
    )
    parser.add_argument(
        "--n_queries", type=int, default=4,
        help="DETR decoder slots. Each color trial has exactly 1 event so 4 "
             "is plenty; the unmatched queries collapse to background.",
    )
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
