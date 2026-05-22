"""Offline item-mux inference + evaluation.

Loads one or more recorded sessions (the same format that `train_item_mux.py`
consumes), runs a trained EEGNetMux model on each cue's active window, and
dumps predictions, an accuracy JSON, and matplotlib figures (per-head
confusion matrices, per-episode timelines, per-session accuracy bars,
probability distributions) to `inference/data/<name>/`.

Example calls (run from the repo root):

    # Single session.
    python inference/inference_item_mux.py \\
        --model training/eegnet_models/ITEM_MUX_500/item_mux_eegnet.pt \\
        --session_ids 2026-05-19_17-14-52 \\
        --name read_react

    # Evaluate across multiple sessions.
    python inference/inference_item_mux.py \\
        --model training/eegnet_models/ITEM_MUX_500/item_mux_eegnet.pt \\
        --session_ids 2026-05-19_17-14-52 2026-05-20_09-12-03 \\
        --name eval_2026-05-20

Sessions must share `active_s` with the model's training sidecar (50ms
tolerance) and use the same item set.
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

from train_item_mux import (  # noqa: E402
    CHANNELS,
    SAMPLING_RATE,
    TIMING_TOLERANCE_MS,
    EEGNetMux,
    infer_timing,
    load_session,
)


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
    items: List[str],
    n_locations: int,
) -> List[dict]:
    """Run the model on every cue in a session. Returns per-cue rows."""
    times, samples, timing, events, _, sess_items = load_session(session_dir)
    if sess_items != items:
        raise ValueError(
            f"{session_dir}: items={sess_items} differ from model items={items}"
        )
    if abs(timing.active_s - expected_active_s) * 1000 > TIMING_TOLERANCE_MS:
        raise ValueError(
            f"{session_dir}: active_s={timing.active_s:.3f} differs from model "
            f"active_s={expected_active_s:.3f} (tolerance {TIMING_TOLERANCE_MS}ms)"
        )

    item_to_id = {name: i for i, name in enumerate(items)}
    id_to_item = {i: name for name, i in item_to_id.items()}
    active_ms = expected_active_s * 1000.0
    rows: List[dict] = []
    for episode_idx, (_, ev) in enumerate(events.iterrows()):
        if ev["event"] != "cue":
            continue
        label = ev["label"]
        if label not in item_to_id:
            continue
        loc = int(ev["location"])
        if loc < 0 or loc >= n_locations:
            continue
        t_start = float(ev["relative_time_ms"])
        mask = (times >= t_start) & (times < t_start + active_ms)
        window = samples[mask]
        if window.shape[0] == 0:
            continue
        x = preprocess(window, target_T)
        with torch.no_grad():
            logits_i, logits_l = model(torch.from_numpy(x).unsqueeze(0).to(device))
            probs_i = torch.softmax(logits_i, dim=1).cpu().numpy()[0]
            probs_l = torch.softmax(logits_l, dim=1).cpu().numpy()[0]
            pred_item_id = int(np.argmax(probs_i))
            pred_loc = int(np.argmax(probs_l))
        pred_item = id_to_item[pred_item_id]
        row = {
            "session_id": os.path.basename(session_dir.rstrip("/")),
            "episode": episode_idx,
            "cue_ms": t_start,
            "item": label,
            "item_id": item_to_id[label],
            "pred_item": pred_item,
            "pred_item_id": pred_item_id,
            "location": loc,
            "pred_location": pred_loc,
            "item_correct": int(pred_item == label),
            "location_correct": int(pred_loc == loc),
            "both_correct": int(pred_item == label and pred_loc == loc),
            "pred_item_prob": float(probs_i[pred_item_id]),
            "pred_location_prob": float(probs_l[pred_loc]),
        }
        for i, name in enumerate(items):
            row[f"prob_item_{name}"] = float(probs_i[i])
        for i in range(n_locations):
            row[f"prob_loc_{i}"] = float(probs_l[i])
        rows.append(row)
    return rows


def compute_accuracy(rows: List[dict], items: List[str], n_locations: int) -> dict:
    total = len(rows)
    item_correct = sum(r["item_correct"] for r in rows)
    loc_correct = sum(r["location_correct"] for r in rows)
    both_correct = sum(r["both_correct"] for r in rows)

    per_item: Dict[str, List[int]] = {name: [0, 0] for name in items}
    per_location: Dict[int, List[int]] = {i: [0, 0] for i in range(n_locations)}
    item_cm = np.zeros((len(items), len(items)), dtype=int)
    loc_cm = np.zeros((n_locations, n_locations), dtype=int)
    per_session: Dict[str, dict] = {}

    item_to_id = {name: i for i, name in enumerate(items)}
    for r in rows:
        per_item[r["item"]][1] += 1
        per_item[r["item"]][0] += r["item_correct"]
        per_location[r["location"]][1] += 1
        per_location[r["location"]][0] += r["location_correct"]
        item_cm[item_to_id[r["item"]], item_to_id[r["pred_item"]]] += 1
        loc_cm[r["location"], r["pred_location"]] += 1
        s = per_session.setdefault(
            r["session_id"], {"item_c": 0, "loc_c": 0, "both_c": 0, "total": 0}
        )
        s["total"] += 1
        s["item_c"] += r["item_correct"]
        s["loc_c"] += r["location_correct"]
        s["both_c"] += r["both_correct"]

    return {
        "total_episodes": total,
        "item_correct": item_correct,
        "location_correct": loc_correct,
        "both_correct": both_correct,
        "item_accuracy": item_correct / total if total else 0.0,
        "location_accuracy": loc_correct / total if total else 0.0,
        "joint_accuracy": both_correct / total if total else 0.0,
        "per_item": {
            name: {"correct": c, "total": t, "accuracy": (c / t) if t else 0.0}
            for name, (c, t) in per_item.items()
        },
        "per_location": {
            str(i): {"correct": c, "total": t, "accuracy": (c / t) if t else 0.0}
            for i, (c, t) in per_location.items()
        },
        "per_session": {
            sid: {
                "total": s["total"],
                "item_accuracy": s["item_c"] / s["total"] if s["total"] else 0.0,
                "location_accuracy": s["loc_c"] / s["total"] if s["total"] else 0.0,
                "joint_accuracy": s["both_c"] / s["total"] if s["total"] else 0.0,
            }
            for sid, s in per_session.items()
        },
        "item_confusion": item_cm.tolist(),
        "location_confusion": loc_cm.tolist(),
        "_item_cm": item_cm,
        "_loc_cm": loc_cm,
        "_items": items,
        "_n_locations": n_locations,
    }


def print_accuracy(summary: dict):
    items = summary["_items"]
    n_loc = summary["_n_locations"]
    print("\n=== Accuracy summary ===")
    print(
        f"Episodes: {summary['total_episodes']}  "
        f"item={summary['item_accuracy']:.3f}  "
        f"location={summary['location_accuracy']:.3f}  "
        f"joint={summary['joint_accuracy']:.3f}"
    )
    print("Per item:")
    for name, stats in summary["per_item"].items():
        print(f"  {name}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.3f}")
    print("Per location:")
    for slot, stats in summary["per_location"].items():
        print(f"  slot {slot}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.3f}")
    if len(summary["per_session"]) > 1:
        print("Per session:")
        for sid, stats in summary["per_session"].items():
            print(
                f"  {sid}: n={stats['total']} "
                f"item={stats['item_accuracy']:.3f} "
                f"loc={stats['location_accuracy']:.3f} "
                f"joint={stats['joint_accuracy']:.3f}"
            )
    print(f"Item confusion (rows=truth, cols=pred), labels={items}:")
    print(summary["_item_cm"])
    print(f"Location confusion (rows=truth, cols=pred), {n_loc} slots:")
    print(summary["_loc_cm"])


def _plot_confusion(matrix: np.ndarray, labels: List[str], title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    n = len(labels)
    ax.set_xticks(range(n), labels=labels)
    ax.set_yticks(range(n), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_item(summary: dict, out_path: str):
    _plot_confusion(
        summary["_item_cm"],
        summary["_items"],
        f"Item confusion (acc={summary['item_accuracy']:.3f})",
        out_path,
    )


def plot_confusion_location(summary: dict, out_path: str):
    n = summary["_n_locations"]
    _plot_confusion(
        summary["_loc_cm"],
        [str(i) for i in range(n)],
        f"Location confusion (acc={summary['location_accuracy']:.3f})",
        out_path,
    )


def _plot_timeline(
    rows: List[dict],
    truth_key: str,
    pred_key: str,
    correct_key: str,
    n_classes: int,
    yticklabels: List[str],
    title: str,
    out_path: str,
):
    n = len(rows)
    x = np.arange(n)
    truth = np.array([r[truth_key] for r in rows])
    pred = np.array([r[pred_key] for r in rows])
    correct = np.array([r[correct_key] for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.25), 3.5))
    ax.scatter(x, truth - 0.08, marker="o", s=40, color="black",
               label="truth", zorder=2)
    ax.scatter(x[correct], pred[correct] + 0.08, marker="x", s=50,
               color="tab:green", label="pred (correct)", zorder=3)
    ax.scatter(x[~correct], pred[~correct] + 0.08, marker="x", s=50,
               color="tab:red", label="pred (wrong)", zorder=3)

    sessions = [r["session_id"] for r in rows]
    for i in range(1, n):
        if sessions[i] != sessions[i - 1]:
            ax.axvline(i - 0.5, color="grey", linestyle="--", linewidth=0.8)

    ax.set_yticks(range(n_classes), labels=yticklabels)
    ax.set_ylim(-0.5, n_classes - 0.5)
    ax.set_xlabel("Episode (across sessions, in order)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeline_item(rows: List[dict], items: List[str], out_path: str):
    _plot_timeline(
        rows, "item_id", "pred_item_id", "item_correct",
        len(items), items, "Item: predictions vs truth", out_path,
    )


def plot_timeline_location(rows: List[dict], n_locations: int, out_path: str):
    _plot_timeline(
        rows, "location", "pred_location", "location_correct",
        n_locations, [str(i) for i in range(n_locations)],
        "Location: predictions vs truth", out_path,
    )


def plot_session_bars(summary: dict, out_path: str):
    sessions = list(summary["per_session"].keys())
    item_accs = [summary["per_session"][s]["item_accuracy"] for s in sessions]
    loc_accs = [summary["per_session"][s]["location_accuracy"] for s in sessions]
    joint_accs = [summary["per_session"][s]["joint_accuracy"] for s in sessions]
    counts = [summary["per_session"][s]["total"] for s in sessions]

    x = np.arange(len(sessions))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(4, len(sessions) * 1.6), 4))
    ax.bar(x - width, item_accs, width, color="tab:blue", label="item")
    ax.bar(x, loc_accs, width, color="tab:cyan", label="location")
    ax.bar(x + width, joint_accs, width, color="tab:purple", label="joint")
    item_chance = 1.0 / max(len(summary["_items"]), 1)
    loc_chance = 1.0 / max(summary["_n_locations"], 1)
    ax.axhline(item_chance, color="grey", linestyle="--", linewidth=0.8,
               label=f"item chance ({item_chance:.2f})")
    if abs(loc_chance - item_chance) > 1e-6:
        ax.axhline(loc_chance, color="grey", linestyle=":", linewidth=0.8,
                   label=f"loc chance ({loc_chance:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-session accuracy")
    for xi, n in zip(x, counts):
        ax.text(xi, 1.0, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_prob_distribution(rows, prob_key, correct_key, title, out_path):
    correct_probs = []
    wrong_probs = []
    for r in rows:
        (correct_probs if r[correct_key] else wrong_probs).append(r[prob_key])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(correct_probs, bins=bins, alpha=0.6, color="tab:green",
            label=f"correct (n={len(correct_probs)})")
    ax.hist(wrong_probs, bins=bins, alpha=0.6, color="tab:red",
            label=f"wrong (n={len(wrong_probs)})")
    ax.set_xlabel("Predicted-class probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prob_distribution_item(rows: List[dict], out_path: str):
    _plot_prob_distribution(
        rows, "pred_item_prob", "item_correct",
        "Item head confidence by outcome", out_path,
    )


def plot_prob_distribution_location(rows: List[dict], out_path: str):
    _plot_prob_distribution(
        rows, "pred_location_prob", "location_correct",
        "Location head confidence by outcome", out_path,
    )


def write_predictions_csv(rows: List[dict], items: List[str], n_locations: int, out_path: str):
    fieldnames = [
        "session_id", "episode", "cue_ms",
        "item", "item_id", "pred_item", "pred_item_id",
        "location", "pred_location",
        "item_correct", "location_correct", "both_correct",
        "pred_item_prob", "pred_location_prob",
    ]
    fieldnames += [f"prob_item_{name}" for name in items]
    fieldnames += [f"prob_loc_{i}" for i in range(n_locations)]
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
    items = list(sidecar["items"])
    n_locations = int(sidecar["n_locations"])
    if sidecar["channels"] != CHANNELS:
        raise ValueError(
            f"Channel mismatch: sidecar={sidecar['channels']} vs CHANNELS={CHANNELS}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetMux(
        n_channels=len(CHANNELS),
        n_samples=target_T,
        n_items=len(items),
        n_locations=n_locations,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(
        f"Loaded {args.model} (active_s={expected_active_s:.3f}, T={target_T}, "
        f"items={items}, n_locations={n_locations}) on {device}"
    )

    out_dir = os.path.join(args.data_root, args.name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    all_rows: List[dict] = []
    for sid in args.session_ids:
        session_dir = os.path.join(args.session_data_root, sid)
        rows = predict_session(
            session_dir, model, device, expected_active_s, target_T,
            items, n_locations,
        )
        print(f"  session {sid}: {len(rows)} predictions")
        all_rows.extend(rows)

    if not all_rows:
        print("No predictions produced; nothing to write.")
        return

    summary = compute_accuracy(all_rows, items, n_locations)
    print_accuracy(summary)

    write_predictions_csv(all_rows, items, n_locations,
                          os.path.join(out_dir, "predictions.csv"))
    summary_to_save = {
        k: v for k, v in summary.items()
        if not k.startswith("_")
    }
    summary_to_save["model"] = args.model
    summary_to_save["session_ids"] = args.session_ids
    summary_to_save["items"] = items
    summary_to_save["n_locations"] = n_locations
    with open(os.path.join(out_dir, "accuracy.json"), "w") as f:
        json.dump(summary_to_save, f, indent=2)

    plot_confusion_item(summary, os.path.join(out_dir, "confusion_item.png"))
    plot_confusion_location(summary, os.path.join(out_dir, "confusion_location.png"))
    plot_timeline_item(all_rows, items, os.path.join(out_dir, "timeline_item.png"))
    plot_timeline_location(all_rows, n_locations,
                           os.path.join(out_dir, "timeline_location.png"))
    plot_session_bars(summary, os.path.join(out_dir, "per_session_accuracy.png"))
    plot_prob_distribution_item(
        all_rows, os.path.join(out_dir, "probability_distribution_item.png")
    )
    plot_prob_distribution_location(
        all_rows, os.path.join(out_dir, "probability_distribution_location.png")
    )
    print(f"Wrote predictions, accuracy, and figures to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Offline item-mux inference + figures.")
    parser.add_argument("--model", required=True,
                        help="Path to .pt file (sidecar .json must be alongside).")
    parser.add_argument("--session_ids", nargs="+", required=True,
                        help="Session directory names under --session_data_root.")
    parser.add_argument("--name", required=True,
                        help="Output subdir name under --data_root.")
    parser.add_argument("--session_data_root", default="data/item_mux",
                        help="Root holding the recorded sessions (default: data/item_mux).")
    parser.add_argument("--data_root", default=os.path.join("inference", "data"),
                        help="Root for inference output (default: inference/data).")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
