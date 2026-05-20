"""Offline LEFT/RIGHT inference + evaluation.

Loads one or more recorded sessions (the same format that `train_lr.py`
consumes), runs a trained EEGNet model on each cue's active window, and dumps
predictions, an accuracy JSON, and matplotlib figures (confusion matrix,
per-episode timeline, per-session accuracy bars, probability distribution)
to `inference/data/<name>/`.

Example calls (run from the repo root):

    # Single session.
    python inference/inference_lr.py \\
        --model training/models/LR_500_SAMPLES/lr_eegnet.pt \\
        --session_ids 2026-05-19_17-14-52 \\
        --name read_react

    # Evaluate across multiple sessions.
    python inference/inference_lr.py \\
        --model training/models/LR_500_SAMPLES/lr_eegnet.pt \\
        --session_ids 2026-05-19_17-14-52 2026-05-20_09-12-03 \\
        --name eval_2026-05-20

Sessions must share `active_s` with the model's training sidecar (50ms
tolerance) so the input window length matches.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))

from train_lr import (  # noqa: E402
    CHANNELS,
    LABEL_TO_ID,
    SAMPLING_RATE,
    TIMING_TOLERANCE_MS,
    EEGNet,
    infer_timing,
    load_session,
)

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def preprocess(window: np.ndarray, target_T: int) -> np.ndarray:
    """Crop/pad to target_T samples and per-channel z-score. Returns (C, T)."""
    if window.shape[0] >= target_T:
        window = window[:target_T]
    else:
        pad = np.zeros((target_T - window.shape[0], window.shape[1]), dtype=np.float32)
        window = np.concatenate([window, pad], axis=0)
    x = window.T.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mean) / std


def predict_session(
    session_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    expected_active_s: float,
    target_T: int,
) -> Tuple[List[dict], dict]:
    """Run the model on every cue in a session. Returns (rows, summary)."""
    times, samples, timing, events, _ = load_session(session_dir)
    if abs(timing.active_s - expected_active_s) * 1000 > TIMING_TOLERANCE_MS:
        raise ValueError(
            f"{session_dir}: active_s={timing.active_s:.3f} differs from model "
            f"active_s={expected_active_s:.3f} (tolerance {TIMING_TOLERANCE_MS}ms)"
        )

    active_ms = expected_active_s * 1000.0
    rows: List[dict] = []
    for episode_idx, (_, ev) in enumerate(events.iterrows()):
        if ev["event"] != "cue":
            continue
        label = ev["label"]
        if label not in LABEL_TO_ID:
            continue
        t_start = float(ev["relative_time_ms"])
        mask = (times >= t_start) & (times < t_start + active_ms)
        window = samples[mask]
        if window.shape[0] == 0:
            continue
        x = preprocess(window, target_T)
        with torch.no_grad():
            logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
        pred_label = ID_TO_LABEL[pred_id]
        rows.append({
            "session_id": os.path.basename(session_dir.rstrip("/")),
            "episode": episode_idx,
            "cue_ms": t_start,
            "label": label,
            "label_id": LABEL_TO_ID[label],
            "prediction": pred_label,
            "prediction_id": pred_id,
            "prob_LEFT": float(probs[LABEL_TO_ID["LEFT"]]),
            "prob_RIGHT": float(probs[LABEL_TO_ID["RIGHT"]]),
            "correct": int(pred_label == label),
        })
    return rows


def compute_accuracy(rows: List[dict]) -> dict:
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    per_class = {"LEFT": [0, 0], "RIGHT": [0, 0]}
    confusion = {("LEFT", "LEFT"): 0, ("LEFT", "RIGHT"): 0,
                 ("RIGHT", "LEFT"): 0, ("RIGHT", "RIGHT"): 0}
    per_session: Dict[str, List[int]] = {}
    for r in rows:
        per_class[r["label"]][1] += 1
        if r["correct"]:
            per_class[r["label"]][0] += 1
        confusion[(r["label"], r["prediction"])] += 1
        per_session.setdefault(r["session_id"], [0, 0])
        per_session[r["session_id"]][1] += 1
        per_session[r["session_id"]][0] += r["correct"]

    return {
        "total_episodes": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_class": {
            cls: {
                "correct": c,
                "total": t,
                "accuracy": (c / t) if t else 0.0,
            }
            for cls, (c, t) in per_class.items()
        },
        "per_session": {
            sid: {
                "correct": c,
                "total": t,
                "accuracy": (c / t) if t else 0.0,
            }
            for sid, (c, t) in per_session.items()
        },
        "confusion_matrix": {
            "rows_are_truth": True,
            "LEFT":  {"LEFT":  confusion[("LEFT",  "LEFT")],
                      "RIGHT": confusion[("LEFT",  "RIGHT")]},
            "RIGHT": {"LEFT":  confusion[("RIGHT", "LEFT")],
                      "RIGHT": confusion[("RIGHT", "RIGHT")]},
        },
        "_confusion_raw": confusion,
    }


def print_accuracy(summary: dict):
    print("\n=== Accuracy summary ===")
    print(f"Episodes: {summary['total_episodes']}, "
          f"correct: {summary['correct']}, "
          f"accuracy: {summary['accuracy']:.3f}")
    for cls, stats in summary["per_class"].items():
        print(f"  {cls}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.3f}")
    if len(summary["per_session"]) > 1:
        print("Per session:")
        for sid, stats in summary["per_session"].items():
            print(f"  {sid}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.3f}")
    c = summary["_confusion_raw"]
    print("Confusion (rows=truth, cols=pred):")
    print(f"          LEFT  RIGHT")
    print(f"  LEFT   {c[('LEFT','LEFT')]:>5} {c[('LEFT','RIGHT')]:>5}")
    print(f"  RIGHT  {c[('RIGHT','LEFT')]:>5} {c[('RIGHT','RIGHT')]:>5}")


def plot_confusion(summary: dict, out_path: str):
    c = summary["_confusion_raw"]
    matrix = np.array([
        [c[("LEFT", "LEFT")],  c[("LEFT", "RIGHT")]],
        [c[("RIGHT", "LEFT")], c[("RIGHT", "RIGHT")]],
    ], dtype=int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["LEFT", "RIGHT"])
    ax.set_yticks([0, 1], labels=["LEFT", "RIGHT"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (acc={summary['accuracy']:.3f})")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeline(rows: List[dict], out_path: str):
    n = len(rows)
    x = np.arange(n)
    truth = np.array([r["label_id"] for r in rows])
    pred = np.array([r["prediction_id"] for r in rows])
    correct = np.array([r["correct"] for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.25), 3.5))
    ax.scatter(x, truth - 0.08, marker="o", s=40, color="black",
               label="truth", zorder=2)
    ax.scatter(x[correct], pred[correct] + 0.08, marker="x", s=50,
               color="tab:green", label="pred (correct)", zorder=3)
    ax.scatter(x[~correct], pred[~correct] + 0.08, marker="x", s=50,
               color="tab:red", label="pred (wrong)", zorder=3)

    # Session boundaries (vertical lines at session changes).
    sessions = [r["session_id"] for r in rows]
    for i in range(1, n):
        if sessions[i] != sessions[i - 1]:
            ax.axvline(i - 0.5, color="grey", linestyle="--", linewidth=0.8)

    ax.set_yticks([0, 1], labels=["LEFT", "RIGHT"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Episode (across sessions, in order)")
    ax.set_title("Per-episode predictions vs truth")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_session_bars(summary: dict, out_path: str):
    sessions = list(summary["per_session"].keys())
    accs = [summary["per_session"][s]["accuracy"] for s in sessions]
    counts = [summary["per_session"][s]["total"] for s in sessions]

    fig, ax = plt.subplots(figsize=(max(4, len(sessions) * 1.2), 4))
    bars = ax.bar(range(len(sessions)), accs, color="tab:blue")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-session accuracy")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prob_distribution(rows: List[dict], out_path: str):
    # Probability assigned to the *predicted* class (model confidence).
    correct_probs = []
    wrong_probs = []
    for r in rows:
        p = r["prob_LEFT"] if r["prediction"] == "LEFT" else r["prob_RIGHT"]
        (correct_probs if r["correct"] else wrong_probs).append(p)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.linspace(0.5, 1.0, 11)
    ax.hist(correct_probs, bins=bins, alpha=0.6, color="tab:green",
            label=f"correct (n={len(correct_probs)})")
    ax.hist(wrong_probs, bins=bins, alpha=0.6, color="tab:red",
            label=f"wrong (n={len(wrong_probs)})")
    ax.set_xlabel("Predicted-class probability")
    ax.set_ylabel("Count")
    ax.set_title("Model confidence by outcome")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_predictions_csv(rows: List[dict], out_path: str):
    fieldnames = [
        "session_id", "episode", "cue_ms",
        "label", "label_id", "prediction", "prediction_id",
        "prob_LEFT", "prob_RIGHT", "correct",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def run(args):
    sidecar_path = os.path.splitext(args.model)[0] + ".json"
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    expected_active_s = float(sidecar["active_s"])
    target_T = int(sidecar["T"])
    if sidecar["channels"] != CHANNELS:
        raise ValueError(
            f"Channel mismatch: sidecar={sidecar['channels']} vs CHANNELS={CHANNELS}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(n_channels=len(CHANNELS), n_samples=target_T).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded {args.model} (active_s={expected_active_s:.3f}, T={target_T}) on {device}")

    out_dir = os.path.join(args.data_root, args.name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    all_rows: List[dict] = []
    for sid in args.session_ids:
        session_dir = os.path.join(args.session_data_root, sid)
        rows = predict_session(session_dir, model, device, expected_active_s, target_T)
        print(f"  session {sid}: {len(rows)} predictions")
        all_rows.extend(rows)

    if not all_rows:
        print("No predictions produced; nothing to write.")
        return

    summary = compute_accuracy(all_rows)
    print_accuracy(summary)

    write_predictions_csv(all_rows, os.path.join(out_dir, "predictions.csv"))
    summary_to_save = {k: v for k, v in summary.items() if k != "_confusion_raw"}
    summary_to_save["model"] = args.model
    summary_to_save["session_ids"] = args.session_ids
    with open(os.path.join(out_dir, "accuracy.json"), "w") as f:
        json.dump(summary_to_save, f, indent=2)

    plot_confusion(summary, os.path.join(out_dir, "confusion_matrix.png"))
    plot_timeline(all_rows, os.path.join(out_dir, "timeline.png"))
    plot_session_bars(summary, os.path.join(out_dir, "per_session_accuracy.png"))
    plot_prob_distribution(all_rows, os.path.join(out_dir, "probability_distribution.png"))
    print(f"Wrote predictions, accuracy, and 4 figures to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Offline LEFT/RIGHT inference + figures.")
    parser.add_argument("--model", required=True,
                        help="Path to .pt file (sidecar .json must be alongside).")
    parser.add_argument("--session_ids", nargs="+", required=True,
                        help="Session directory names under --session_data_root.")
    parser.add_argument("--name", required=True,
                        help="Output subdir name under --data_root.")
    parser.add_argument("--session_data_root", default="data",
                        help="Root holding the recorded sessions (default: data).")
    parser.add_argument("--data_root", default=os.path.join("inference", "data"),
                        help="Root for inference output (default: inference/data).")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
