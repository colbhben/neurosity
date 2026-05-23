"""Offline inference for the two-head HITL EEGNet.

Loads a checkpoint trained by `training/train_clport_eegnet.py`, slices each
session's episodes into the same goal-shown -> place-done windows, predicts
(block_color, bowl_color), and writes per-split metrics + confusion
matrices.

Example:
    python inference/inference_clport_eegnet.py \\
        --model_dir training/eegnet_models/clport_eegnet_v0 \\
        --session_dirs data/clport/2026-05-22_10-00-00 \\
        --out_dir inference_runs/clport_eegnet_v0
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.clport.session_io import (  # noqa: E402
    CHANNELS, SAMPLING_RATE, build_xy, discover_sessions,
)
from training.clport.splits import (  # noqa: E402
    Episode, HITL_COLORS, color_to_idx, make_splits,
)
from training.train_clport_eegnet import EEGNetTwoHead  # noqa: E402


def _confusion(truth: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    return cm


def _save_confusion_plot(cm: np.ndarray, labels, title: str, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="training/eegnet_models/<name>/")
    parser.add_argument("--session_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--success_only", action="store_true")
    args = parser.parse_args()

    sidecar_path = os.path.join(args.model_dir, "clport_eegnet.json")
    weights_path = os.path.join(args.model_dir, "clport_eegnet.pt")
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    target_T = int(sidecar["T"])

    hitl_eps = discover_sessions(args.session_dirs)
    if args.success_only:
        hitl_eps = [e for e in hitl_eps if e.success]
    if not hitl_eps:
        raise RuntimeError("No episodes discovered.")

    split_eps = [
        Episode(session_id=e.session_id, episode_idx=e.episode_idx,
                block_color=e.block_color, bowl_color=e.bowl_color)
        for e in hitl_eps
    ]
    splits = make_splits(split_eps, seed=0)
    by_key = {(e.session_id, e.episode_idx): e for e in hitl_eps}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetTwoHead(
        n_channels=len(CHANNELS), n_samples=target_T,
        n_classes=len(HITL_COLORS),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    pred_rows = []
    for split_name in ("train", "val_seen", "val_unseen", "val_mixed"):
        ep_list = [by_key[(s.session_id, s.episode_idx)] for s in splits[split_name]]
        if not ep_list:
            summary[split_name] = {"n": 0}
            continue
        X, yb, yw = build_xy(ep_list, target_T)
        with torch.no_grad():
            block_logits, bowl_logits = model(torch.from_numpy(X).to(device))
            block_pred = block_logits.argmax(1).cpu().numpy()
            bowl_pred = bowl_logits.argmax(1).cpu().numpy()
        block_acc = float((block_pred == yb).mean())
        bowl_acc = float((bowl_pred == yw).mean())
        joint_acc = float(((block_pred == yb) & (bowl_pred == yw)).mean())
        summary[split_name] = {
            "n": int(len(ep_list)),
            "block_acc": block_acc,
            "bowl_acc": bowl_acc,
            "joint_acc": joint_acc,
        }
        cm_block = _confusion(yb, block_pred, len(HITL_COLORS))
        cm_bowl = _confusion(yw, bowl_pred, len(HITL_COLORS))
        _save_confusion_plot(
            cm_block, HITL_COLORS, f"Block confusion ({split_name})",
            os.path.join(args.out_dir, f"confusion_block_{split_name}.png"))
        _save_confusion_plot(
            cm_bowl, HITL_COLORS, f"Bowl confusion ({split_name})",
            os.path.join(args.out_dir, f"confusion_bowl_{split_name}.png"))
        for ep, b_t, b_p, w_t, w_p in zip(ep_list, yb, block_pred, yw, bowl_pred):
            pred_rows.append({
                "split": split_name,
                "session_id": ep.session_id,
                "episode_idx": ep.episode_idx,
                "true_block": HITL_COLORS[b_t],
                "pred_block": HITL_COLORS[b_p],
                "true_bowl": HITL_COLORS[w_t],
                "pred_bowl": HITL_COLORS[w_p],
                "joint_correct": bool(b_t == b_p and w_t == w_p),
            })

    pd.DataFrame(pred_rows).to_csv(
        os.path.join(args.out_dir, "predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, "accuracy.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
