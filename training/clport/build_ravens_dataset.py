"""Convert HITL CLIPort sessions into RavensDataset-format pickles.

Reads `data/clport/<session_id>/episodes/*/{obs,action,info}.pkl` plus the
session's `events.csv` + `meta.json`, partitions episodes into the
seen/unseen splits, and emits the directory tree the upstream
`cliport.dataset.RavensDataset` expects:

    <out_root>/<task_name>-train/{color,depth,action,reward,info}/<idx>-<seed>.pkl
    <out_root>/<task_name>-val/...
    <out_root>/<task_name>-val-unseen/...

The `train` and `val` directories together form the seen-composition
population (val_seen). `val-unseen` holds episodes whose (block, bowl) tuple
is in HOLDOUT_UNSEEN.

Optionally writes a parallel `eeg.pkl` per episode containing the
(8, T) z-scored goal-shown -> place-done EEG window so the
EEG-conditioned agent can pick it up by filename.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.clport.session_io import (  # noqa: E402
    SAMPLING_RATE, discover_sessions, slice_eeg_window,
)
from training.clport.splits import (  # noqa: E402
    Episode, make_splits,
)


def _dump(field_path: str, fname: str, data):
    os.makedirs(field_path, exist_ok=True)
    with open(os.path.join(field_path, fname), "wb") as f:
        pickle.dump(data, f)


def _emit(out_dir: str, hitl_ep, idx: int, with_eeg: bool, eeg_T: int):
    """Write one episode in RavensDataset format under `out_dir`."""
    with open(os.path.join(hitl_ep.ep_dir, "obs.pkl"), "rb") as f:
        obs_list = pickle.load(f)
    with open(os.path.join(hitl_ep.ep_dir, "action.pkl"), "rb") as f:
        act_list = pickle.load(f)
    with open(os.path.join(hitl_ep.ep_dir, "info.pkl"), "rb") as f:
        info_list = pickle.load(f)

    color = np.uint8([obs["color"] for obs in obs_list])
    depth = np.float32([obs["depth"] for obs in obs_list])
    # CLIPort stores reward as the cumulative-progress-at-step list; we
    # record the success indicator as a simple monotonic 0/1 step list.
    success = info_list[-1].get("lang_goal", "") if info_list else ""
    reward = [0.0] * (len(obs_list) - 1) + [
        1.0 if hitl_ep.success else 0.0,
    ]
    if len(reward) != len(obs_list):
        reward = [0.0] * len(obs_list)
        if reward:
            reward[-1] = 1.0 if hitl_ep.success else 0.0

    fname = f"{idx:06d}-{hitl_ep.seed}.pkl"
    _dump(os.path.join(out_dir, "color"), fname, color)
    _dump(os.path.join(out_dir, "depth"), fname, depth)
    _dump(os.path.join(out_dir, "action"), fname, act_list)
    _dump(os.path.join(out_dir, "reward"), fname, reward)
    _dump(os.path.join(out_dir, "info"), fname, info_list)

    if with_eeg:
        eeg = slice_eeg_window(
            hitl_ep.eeg_dir, hitl_ep.goal_shown_ms,
            hitl_ep.place_done_ms, eeg_T,
        )
        _dump(os.path.join(out_dir, "eeg"), fname, eeg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dirs", nargs="+", required=True)
    parser.add_argument("--out_root", required=True,
                        help="Destination data root, e.g. data/clport_ravens")
    parser.add_argument("--task_name", default="put-block-in-bowl-hitl")
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with_eeg", action="store_true",
                        help="Also emit eeg.pkl per episode (8 x T window).")
    parser.add_argument("--eeg_window_seconds", type=float, default=8.0)
    args = parser.parse_args()

    eps = discover_sessions(args.session_dirs)
    if args.success_only:
        eps = [e for e in eps if e.success]
    if not eps:
        raise RuntimeError("No episodes discovered.")

    split_eps = [
        Episode(session_id=e.session_id, episode_idx=e.episode_idx,
                block_color=e.block_color, bowl_color=e.bowl_color)
        for e in eps
    ]
    splits = make_splits(split_eps, seed=args.seed)
    by_key = {(e.session_id, e.episode_idx): e for e in eps}

    target_T = int(round(args.eeg_window_seconds * SAMPLING_RATE))

    # CLIPort expects "<task>-train" and "<task>-val" directories.
    # We additionally emit "<task>-val-unseen" for the held-out compositions.
    train_dir = os.path.join(args.out_root, f"{args.task_name}-train")
    val_dir = os.path.join(args.out_root, f"{args.task_name}-val")
    unseen_dir = os.path.join(args.out_root, f"{args.task_name}-val-unseen")

    counts = {}
    for split_name, out_dir in [
        ("train", train_dir), ("val_seen", val_dir), ("val_unseen", unseen_dir),
    ]:
        ep_list = [by_key[(s.session_id, s.episode_idx)] for s in splits[split_name]]
        for i, hitl_ep in enumerate(ep_list):
            _emit(out_dir, hitl_ep, i, args.with_eeg, target_T)
        counts[split_name] = len(ep_list)

    # Emit a manifest so the policy training script can pick up split paths
    # and HITL metadata without re-discovering.
    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump({
            "task_name": args.task_name,
            "counts": counts,
            "with_eeg": args.with_eeg,
            "eeg_window_seconds": args.eeg_window_seconds,
            "eeg_T": target_T,
            "session_dirs": args.session_dirs,
        }, f, indent=2)

    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
