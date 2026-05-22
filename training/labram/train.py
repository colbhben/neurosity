"""LaBraM finetuning trainer.

Example::

    python -m training.labram.train \\
        --name labram_lr \\
        --device_config training/labram/configs/neurosity_crown.yaml \\
        --task_config   training/labram/configs/task_lr.yaml \\
        --session_ids 2026-05-20_09-52-51 2026-05-20_09-55-42 \\
        --epochs 50 --batch_size 16 \\
        --pretrained_ckpt third_party/labram/checkpoints/labram-base.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from training.labram.config import (
    HeadConfig,
    TaskConfig,
    DeviceConfig,
    load_device_config,
    load_task_config,
)
from training.labram.dataset import LabramDataset, collate
from training.labram.model import LaBraMFinetune, load_pretrained
from training.labram.preprocess import build_cache


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _layer_decay_param_groups(
    model: LaBraMFinetune,
    *,
    base_lr: float,
    head_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> List[dict]:
    """Per-layer LR decay across the LaBraM backbone, full LR for heads.

    Mirrors the recipe in `third_party/labram/optim_factory.py` but is kept
    self-contained so it works without importing deepspeed.
    """
    backbone = model.backbone
    n_layers = backbone.get_num_layers() if hasattr(backbone, "get_num_layers") else 12
    no_decay = set(backbone.no_weight_decay()) if hasattr(backbone, "no_weight_decay") else set()

    def layer_id(name: str) -> int:
        if name.startswith("patch_embed") or name in ("cls_token", "pos_embed", "time_embed"):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return n_layers + 1

    groups: Dict[str, dict] = {}
    for n, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(n)
        scale = layer_decay ** (n_layers + 1 - lid)
        wd = 0.0 if (n in no_decay or n.endswith(".bias")) else weight_decay
        key = f"backbone_layer_{lid}_wd{wd}"
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * scale, "weight_decay": wd}
        groups[key]["params"].append(p)

    head_params = [p for n, p in model.heads.named_parameters() if p.requires_grad]
    if head_params:
        groups["heads"] = {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    return list(groups.values())


def _cosine_lr(base_lrs: List[float], step: int, total: int, warmup: int) -> List[float]:
    """Cosine schedule with linear warmup. Returns lr for each param group."""
    if step < warmup:
        scale = step / max(1, warmup)
    else:
        prog = (step - warmup) / max(1, total - warmup)
        scale = 0.5 * (1 + math.cos(math.pi * prog))
    return [lr * scale for lr in base_lrs]


def _split_indices(n: int, val_frac: float, seed: int, max_trials: Optional[int]):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    if max_trials is not None and max_trials < len(idx):
        idx = idx[:max_trials]
    cut = max(1, int(round((1 - val_frac) * len(idx))))
    return idx[:cut].tolist(), idx[cut:].tolist()


def _build_optimizer(model, args):
    groups = _layer_decay_param_groups(
        model,
        base_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
    )
    base_lrs = [g["lr"] for g in groups]
    optim = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    return optim, base_lrs


def _temporal_shift(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Random temporal roll along the time axis (last dim of (B,C,A,P) view)."""
    if max_shift <= 0:
        return x
    B, C, A, P = x.shape
    flat = x.reshape(B, C, A * P)
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    out = torch.empty_like(flat)
    for i in range(B):
        out[i] = torch.roll(flat[i], shifts=int(shifts[i].item()), dims=-1)
    return out.reshape(B, C, A, P)


def train(args):
    _set_seed(args.seed)

    device_cfg: DeviceConfig = load_device_config(args.device_config)
    task_cfg: TaskConfig = load_task_config(args.task_config)
    if not task_cfg.heads:
        raise ValueError("task config has no heads defined")

    print(f"[1/4] Building cache for {len(args.session_ids)} session(s)...")
    cache_dirs = build_cache(
        args.session_ids,
        device_cfg,
        task_cfg,
        cache_root=args.cache_root,
        force=args.force_preprocess,
    )

    head_names = [h.name for h in task_cfg.heads]
    ds = LabramDataset(cache_dirs, head_names=head_names)
    if len(ds) == 0:
        raise RuntimeError("Cache produced 0 trials.")
    print(f"[2/4] Cache trials: {len(ds)}, shape per trial: {ds.shape[1:]}")

    train_idx, val_idx = _split_indices(len(ds), args.val_frac, args.seed, args.max_trials)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if val_idx else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        collate_fn=collate, num_workers=args.num_workers,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers,
        )
        if val_ds is not None
        else None
    )

    print("[3/4] Building model...")
    model = LaBraMFinetune(
        channels=device_cfg.channels,
        heads=task_cfg.heads,
        backbone=args.backbone,
        drop_path_rate=args.drop_path,
    )
    if args.pretrained_ckpt:
        load_pretrained(model, args.pretrained_ckpt)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    if args.unfreeze_last_n_blocks > 0:
        # Freeze everything, then re-enable the top N transformer blocks +
        # the final norm/fc_norm. Heads remain trainable.
        for p in model.backbone.parameters():
            p.requires_grad = False
        n_blocks = len(model.backbone.blocks)
        unfreeze = list(range(max(0, n_blocks - args.unfreeze_last_n_blocks), n_blocks))
        for i in unfreeze:
            for p in model.backbone.blocks[i].parameters():
                p.requires_grad = True
        if hasattr(model.backbone, "norm"):
            for p in model.backbone.norm.parameters():
                p.requires_grad = True
        if getattr(model.backbone, "fc_norm", None) is not None:
            for p in model.backbone.fc_norm.parameters():
                p.requires_grad = True
        print(f"  partial unfreeze: blocks {unfreeze} + norm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,} ; device: {device}")

    optim, base_lrs = _build_optimizer(model, args)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    print("[4/4] Training...")
    head_weights = {h.name: h.weight for h in task_cfg.heads}
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    for h in task_cfg.heads:
        history[f"train_loss_{h.name}"] = []
        history[f"val_loss_{h.name}"] = []
        history[f"train_acc_{h.name}"] = []
        history[f"val_acc_{h.name}"] = []
    head_specs = {h.name: h for h in task_cfg.heads}

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    global_step = 0

    def _accumulate_correct(running, name, head, outputs, targets, masks):
        """Update running['correct'][name], running['total'][name] for classify/grid heads."""
        if head.type not in ("classify", "grid"):
            return
        with torch.no_grad():
            if head.type == "classify":
                pred = outputs[name].argmax(-1)
                ok = (pred == targets[name]) & masks[name]
            else:
                logits = outputs[name]
                pred = logits.argmax(-1)
                ok = (pred == targets[name]) & masks[name]
            running["correct"][name] += int(ok.sum().item())
            running["total"][name] += int(masks[name].sum().item())

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_per_head = {h.name: 0.0 for h in task_cfg.heads}
        train_acc = {"correct": {h.name: 0 for h in task_cfg.heads},
                     "total": {h.name: 0 for h in task_cfg.heads}}
        n_batches = 0
        for batch in train_loader:
            lrs = _cosine_lr(base_lrs, global_step, total_steps, warmup_steps)
            for g, lr in zip(optim.param_groups, lrs):
                g["lr"] = lr
            x = batch.x.to(device)
            if args.temporal_shift:
                x = _temporal_shift(x, args.temporal_shift)
            targets = {k: v.to(device) for k, v in batch.labels.items()}
            masks = {k: v.to(device) for k, v in batch.masks.items()}
            outputs = model(x, targets=targets)
            losses = model.loss(outputs, targets, masks)
            loss = sum(head_weights[k] * losses[k] for k in losses)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            train_total += float(loss.item())
            for k, v in losses.items():
                train_per_head[k] += float(v.item())
            for h in task_cfg.heads:
                _accumulate_correct(train_acc, h.name, h, outputs, targets, masks)
            n_batches += 1
            global_step += 1

        train_total /= max(1, n_batches)
        for k in train_per_head:
            train_per_head[k] /= max(1, n_batches)
        train_acc_per_head = {
            n: (train_acc["correct"][n] / max(1, train_acc["total"][n]))
            if head_specs[n].type in ("classify", "grid") else float("nan")
            for n in train_per_head
        }

        val_total = float("nan")
        val_per_head = {h.name: float("nan") for h in task_cfg.heads}
        val_acc_per_head = {h.name: float("nan") for h in task_cfg.heads}
        if val_loader is not None:
            model.eval()
            val_acc = {"correct": {h.name: 0 for h in task_cfg.heads},
                       "total": {h.name: 0 for h in task_cfg.heads}}
            with torch.no_grad():
                v_total = 0.0
                v_per = {h.name: 0.0 for h in task_cfg.heads}
                v_batches = 0
                for batch in val_loader:
                    x = batch.x.to(device)
                    targets = {k: v.to(device) for k, v in batch.labels.items()}
                    masks = {k: v.to(device) for k, v in batch.masks.items()}
                    outputs = model(x, targets=targets)
                    losses = model.loss(outputs, targets, masks)
                    v_total += float(sum(head_weights[k] * losses[k] for k in losses).item())
                    for k, vv in losses.items():
                        v_per[k] += float(vv.item())
                    for h in task_cfg.heads:
                        _accumulate_correct(val_acc, h.name, h, outputs, targets, masks)
                    v_batches += 1
                val_total = v_total / max(1, v_batches)
                val_per_head = {k: v / max(1, v_batches) for k, v in v_per.items()}
                val_acc_per_head = {
                    n: (val_acc["correct"][n] / max(1, val_acc["total"][n]))
                    if head_specs[n].type in ("classify", "grid") else float("nan")
                    for n in val_per_head
                }

        history["epoch"].append(epoch)
        history["train_loss"].append(train_total)
        history["val_loss"].append(val_total)
        for k in train_per_head:
            history[f"train_loss_{k}"].append(train_per_head[k])
            history[f"val_loss_{k}"].append(val_per_head[k])
            history[f"train_acc_{k}"].append(train_acc_per_head[k])
            history[f"val_acc_{k}"].append(val_acc_per_head[k])

        head_strs = []
        for k in train_per_head:
            s = f"{k}=L{train_per_head[k]:.3f}/{val_per_head[k]:.3f}"
            if head_specs[k].type in ("classify", "grid"):
                s += f" A{train_acc_per_head[k]:.3f}/{val_acc_per_head[k]:.3f}"
            head_strs.append(s)
        print(
            f"  epoch {epoch:3d}  loss={train_total:.4f}/{val_total:.4f}  "
            + "  ".join(head_strs)
        )

        target_metric = val_total if not math.isnan(val_total) else train_total
        if target_metric < best_val:
            best_val = target_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    model.load_state_dict(best_state)

    out_dir = os.path.join("training", "labram_models", args.name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "labram.pt")
    torch.save(model.state_dict(), model_path)

    sidecar = {
        "device_config": asdict(device_cfg),
        "task_config": {
            "name": task_cfg.name,
            "data_root": task_cfg.data_root,
            "window_start_ms": task_cfg.window_start_ms,
            "window_end_ms": task_cfg.window_end_ms,
            "window_seconds": task_cfg.window_seconds,
            "heads": [asdict(h) for h in task_cfg.heads],
        },
        "backbone": args.backbone,
        "embed_dim": int(model.embed_dim),
        "session_ids": args.session_ids,
        "channels": device_cfg.channels,
        "patch_size": 200,
        "trial_shape": list(ds.shape[1:]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
    }
    with open(os.path.join(out_dir, "sidecar.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    train_params = {
        **vars(args),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "n_trials": int(len(ds)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "device": str(device),
    }
    with open(os.path.join(out_dir, "train_params.json"), "w") as f:
        json.dump(train_params, f, indent=2)

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # Final-model predictions on each split for confusion matrices + stats.
    final_stats = _final_split_stats(
        model, device, task_cfg, head_specs, train_loader, val_loader,
    )
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump(final_stats["metrics"], f, indent=2)

    _save_plots(out_dir, history, task_cfg, final_stats)
    print(f"Saved -> {out_dir}")


def _final_split_stats(model, device, task, head_specs, train_loader, val_loader):
    """Run the best-loaded model on each split, gathering preds + stats per head."""
    model.eval()
    splits = {"train": train_loader}
    if val_loader is not None:
        splits["val"] = val_loader

    by_split: dict = {}
    metrics: dict = {}
    for split_name, loader in splits.items():
        head_buf = {h.name: {"pred": [], "target": [], "mask": []} for h in task.heads}
        with torch.no_grad():
            for batch in loader:
                x = batch.x.to(device)
                targets = {k: v.to(device) for k, v in batch.labels.items()}
                masks = {k: v.to(device) for k, v in batch.masks.items()}
                outputs = model(x, targets=targets)
                for h in task.heads:
                    if h.type in ("classify", "grid"):
                        pred = outputs[h.name].argmax(-1)
                    elif h.type == "regress":
                        pred = outputs[h.name]
                    elif h.type == "token":
                        pred = outputs[h.name].argmax(-1)
                    head_buf[h.name]["pred"].append(pred.cpu().numpy())
                    head_buf[h.name]["target"].append(targets[h.name].cpu().numpy())
                    head_buf[h.name]["mask"].append(masks[h.name].cpu().numpy())
        for name, buf in head_buf.items():
            buf["pred"] = np.concatenate(buf["pred"]) if buf["pred"] else np.array([])
            buf["target"] = np.concatenate(buf["target"]) if buf["target"] else np.array([])
            buf["mask"] = np.concatenate(buf["mask"]) if buf["mask"] else np.array([])
        by_split[split_name] = head_buf

        metrics[split_name] = {}
        for h in task.heads:
            buf = head_buf[h.name]
            mask = buf["mask"].astype(bool)
            n = int(mask.sum())
            stats: dict = {"n": n}
            if h.type in ("classify", "grid"):
                pred = buf["pred"][mask]
                target = buf["target"][mask]
                if n == 0:
                    stats.update({"acc": float("nan")})
                else:
                    acc = float((pred == target).mean())
                    stats["acc"] = acc
                    n_classes = (
                        h.num_classes if h.type == "classify"
                        else h.rows * h.cols
                    )
                    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
                    for t, p in zip(target.astype(int), pred.astype(int)):
                        if 0 <= t < n_classes and 0 <= p < n_classes:
                            cm[t, p] += 1
                    # Macro precision / recall / F1.
                    prec = np.zeros(n_classes)
                    rec = np.zeros(n_classes)
                    for c in range(n_classes):
                        tp = cm[c, c]
                        prec[c] = tp / max(1, cm[:, c].sum())
                        rec[c] = tp / max(1, cm[c, :].sum())
                    f1 = np.where(
                        (prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0
                    )
                    stats["confusion_matrix"] = cm.tolist()
                    stats["macro_precision"] = float(prec.mean())
                    stats["macro_recall"] = float(rec.mean())
                    stats["macro_f1"] = float(f1.mean())
                    stats["per_class_f1"] = f1.tolist()
            elif h.type == "regress":
                pred = buf["pred"][mask]
                target = buf["target"][mask]
                if n == 0:
                    stats.update({"mse": float("nan"), "mae": float("nan")})
                else:
                    err = pred - target
                    stats["mse"] = float((err ** 2).mean())
                    stats["mae"] = float(np.abs(err).mean())
            metrics[split_name][h.name] = stats
    return {"by_split": by_split, "metrics": metrics}


def _save_plots(out_dir: str, history: dict, task: TaskConfig, final_stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    has_val = any(not math.isnan(v) for v in history["val_loss"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="train", color="tab:blue")
    if has_val:
        ax.plot(epochs, history["val_loss"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted loss")
    ax.set_title(f"{task.name} loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in task.heads:
        ax.plot(epochs, history[f"train_loss_{h.name}"], label=f"{h.name} train")
        if any(not math.isnan(v) for v in history[f"val_loss_{h.name}"]):
            ax.plot(epochs, history[f"val_loss_{h.name}"], "--", label=f"{h.name} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("per-head loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_head_loss.png"), dpi=120)
    plt.close(fig)

    classify_heads = [h for h in task.heads if h.type in ("classify", "grid")]
    if classify_heads:
        fig, ax = plt.subplots(figsize=(7, 4))
        for h in classify_heads:
            ax.plot(epochs, history[f"train_acc_{h.name}"], label=f"{h.name} train")
            if any(not math.isnan(v) for v in history[f"val_acc_{h.name}"]):
                ax.plot(epochs, history[f"val_acc_{h.name}"], "--", label=f"{h.name} val")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (binary)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("per-head accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "accuracy.png"), dpi=120)
        plt.close(fig)

    # Confusion matrices per head, per split (best-model snapshot).
    for split_name in ("train", "val"):
        if split_name not in final_stats["metrics"]:
            continue
        for h in classify_heads:
            stats = final_stats["metrics"][split_name][h.name]
            cm = np.asarray(stats.get("confusion_matrix", []))
            if cm.size == 0:
                continue
            n_classes = cm.shape[0]
            if h.type == "classify" and h.classes:
                labels = h.classes
            else:
                labels = [str(i) for i in range(n_classes)]
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(labels, rotation=45 if n_classes > 4 else 0, ha="right")
            ax.set_yticklabels(labels)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            acc = stats.get("acc", float("nan"))
            f1 = stats.get("macro_f1", float("nan"))
            ax.set_title(
                f"{h.name} {split_name} (acc={acc:.3f}, F1={f1:.3f}, n={stats['n']})"
            )
            thresh = cm.max() / 2 if cm.max() > 0 else 0
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"confusion_{h.name}_{split_name}.png"), dpi=120
            )
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Finetune LaBraM on Neurosity sessions")
    parser.add_argument("--name", required=True)
    parser.add_argument("--device_config", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--session_ids", nargs="+", required=True)
    parser.add_argument("--cache_root", default="data/labram_cache")
    parser.add_argument("--force_preprocess", action="store_true")

    parser.add_argument("--backbone", default="labram_base_patch200_200")
    parser.add_argument("--pretrained_ckpt", default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=0,
                        help="Freeze backbone except top N transformer blocks + final norm. "
                        "Ignored when --freeze_backbone is set.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone_lr", type=float, default=5e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--layer_decay", type=float, default=0.65)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--temporal_shift", type=int, default=0,
                        help="Random temporal roll in samples (~50ms = 10 at 200Hz).")

    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
