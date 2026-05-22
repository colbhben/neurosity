"""Offline evaluation against held-out sessions.

Loads a model + sidecar saved by `train.py`, builds a cache for the given
sessions, runs inference, and prints per-head metrics. For classify/grid,
reports accuracy + confusion matrix; for regress, reports MSE / MAE; for
token, reports per-token accuracy and a few decoded samples.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.labram.config import (
    DeviceConfig,
    TaskConfig,
    HeadConfig,
)
from training.labram.dataset import LabramDataset, collate
from training.labram.model import LaBraMFinetune
from training.labram.preprocess import build_cache


def _hydrate(sidecar_path: str):
    with open(sidecar_path) as f:
        side = json.load(f)
    device = DeviceConfig.from_dict(side["device_config"])
    tc = side["task_config"]
    heads = [HeadConfig(**h) for h in tc["heads"]]
    task = TaskConfig(
        name=tc["name"],
        data_root=tc["data_root"],
        window_start_ms=tc.get("window_start_ms", 0.0),
        window_end_ms=tc.get("window_end_ms"),
        window_seconds=tc.get("window_seconds"),
        heads=heads,
    )
    return device, task, side


def evaluate(args):
    sidecar_path = os.path.join("training", "labram_models", args.name, "sidecar.json")
    model_path = os.path.join("training", "labram_models", args.name, "labram.pt")
    device_cfg, task, side = _hydrate(sidecar_path)
    if args.data_root:
        task.data_root = args.data_root

    cache_dirs = build_cache(args.session_ids, device_cfg, task, cache_root=args.cache_root)
    head_names = [h.name for h in task.heads]
    ds = LabramDataset(cache_dirs, head_names=head_names)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = LaBraMFinetune(channels=device_cfg.channels, heads=task.heads,
                           backbone=side["backbone"])
    sd = torch.load(model_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds_all: Dict[str, List[np.ndarray]] = {n: [] for n in head_names}
    targets_all: Dict[str, List[np.ndarray]] = {n: [] for n in head_names}
    masks_all: Dict[str, List[np.ndarray]] = {n: [] for n in head_names}
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            preds = model.predict(x)
            for n in head_names:
                preds_all[n].append(preds[n].cpu().numpy())
                targets_all[n].append(batch.labels[n].cpu().numpy())
                masks_all[n].append(batch.masks[n].cpu().numpy())

    print(f"\n{args.name} :: {len(ds)} trials across {len(args.session_ids)} session(s)")
    head_specs = {h.name: h for h in task.heads}
    for n in head_names:
        spec = head_specs[n]
        preds = np.concatenate(preds_all[n])
        target = np.concatenate(targets_all[n])
        mask = np.concatenate(masks_all[n])
        if spec.type == "classify":
            ok = (preds == target) & mask
            print(f"  [{n}] classify acc = {ok.sum() / max(1, mask.sum()):.3f} (n={int(mask.sum())})")
        elif spec.type == "grid":
            r_pred, c_pred = preds[:, 0], preds[:, 1]
            r_t = target // spec.cols
            c_t = target % spec.cols
            ok = ((r_pred == r_t) & (c_pred == c_t)) & mask
            print(f"  [{n}] grid acc = {ok.sum() / max(1, mask.sum()):.3f} (n={int(mask.sum())})")
        elif spec.type == "regress":
            err = preds - target
            sel = mask
            mse = float((err[sel] ** 2).mean()) if sel.any() else float("nan")
            mae = float(np.abs(err[sel]).mean()) if sel.any() else float("nan")
            print(f"  [{n}] regress MSE={mse:.4f} MAE={mae:.4f} (n={int(mask.sum())})")
        elif spec.type == "token":
            tok_ok = (preds[:, : target.shape[1]] == target) & (target != -1)
            denom = (target != -1).sum()
            print(f"  [{n}] token acc = {tok_ok.sum() / max(1, denom):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LaBraM finetuning run")
    parser.add_argument("--name", required=True, help="Run name under training/labram_models/")
    parser.add_argument("--session_ids", nargs="+", required=True)
    parser.add_argument("--data_root", default=None,
                        help="Override task.data_root (e.g. data/lr/react_left_holdout)")
    parser.add_argument("--cache_root", default="data/labram_cache")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
