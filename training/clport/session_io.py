"""Load HITL CLIPort sessions and slice per-episode EEG windows.

Each session lives at `data/clport/<session_id>/` and contains:
    data.csv             -- EEG (Logger format, 256 Hz raw spine)
    events.csv           -- per-event timestamps shared with data.csv
    episodes/<id>-<seed>/meta.json  -- block_color, bowl_color, success, ...

The "goal-text window" used by P(goal_text | EEG) spans from the
`goal_shown` event to the `place_done` event (i.e. the entire scene the
human watched). Windows are zero-padded to a fixed sample count so they
fit a fixed-shape EEGNet input.
"""

from dataclasses import dataclass
from typing import List, Tuple
import json
import os

import numpy as np
import pandas as pd


# Match Logger's channel order; mirrored in workspace/log.py and
# training/train_lr.py to keep CSV column ordering consistent.
CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
SAMPLING_RATE = 256


@dataclass
class HitlEpisode:
    session_id: str
    episode_idx: int
    seed: int
    block_color: str
    bowl_color: str
    success: bool
    pair_pool: str
    goal_shown_ms: float
    place_done_ms: float
    eeg_dir: str  # session_dir for re-loading data.csv
    ep_dir: str   # episodes/<idx>-<seed>/ for obs/action/info/video

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.block_color, self.bowl_color)


def discover_episodes(session_dir: str) -> List[HitlEpisode]:
    """Scan a single session and return one HitlEpisode per episode."""
    events_path = os.path.join(session_dir, "events.csv")
    if not os.path.exists(events_path):
        raise FileNotFoundError(events_path)

    events = pd.read_csv(events_path)
    session_id = os.path.basename(os.path.normpath(session_dir))
    episodes_dir = os.path.join(session_dir, "episodes")

    out: List[HitlEpisode] = []
    for ep_idx in sorted(events["episode"].unique()):
        rows = events[events["episode"] == ep_idx]
        goal = rows[rows["event"] == "goal_shown"]
        place = rows[rows["event"] == "place_done"]
        if len(goal) == 0 or len(place) == 0:
            # Episode was interrupted; skip.
            continue

        seed = int(rows.iloc[0]["seed"]) if "seed" in rows.columns else -1
        ep_dir = os.path.join(episodes_dir, f"{int(ep_idx):06d}-{seed}")
        meta_path = os.path.join(ep_dir, "meta.json")
        if not os.path.exists(meta_path):
            # Fall back to the events.csv row metadata.
            block_color = goal.iloc[0]["block_color"]
            bowl_color = goal.iloc[0]["bowl_color"]
            success = False
            pair_pool = goal.iloc[0].get("pair_pool", "")
        else:
            with open(meta_path) as f:
                meta = json.load(f)
            block_color = meta["block_color"]
            bowl_color = meta["bowl_color"]
            success = bool(meta.get("success", False))
            pair_pool = meta.get("pair_pool", "")

        out.append(HitlEpisode(
            session_id=session_id,
            episode_idx=int(ep_idx),
            seed=seed,
            block_color=block_color,
            bowl_color=bowl_color,
            success=success,
            pair_pool=pair_pool,
            goal_shown_ms=float(goal.iloc[0]["relative_time_ms"]),
            place_done_ms=float(place.iloc[0]["relative_time_ms"]),
            eeg_dir=session_dir,
            ep_dir=ep_dir,
        ))
    return out


def discover_sessions(session_paths: List[str]) -> List[HitlEpisode]:
    """Discover every episode across a list of session directories."""
    episodes: List[HitlEpisode] = []
    for sp in session_paths:
        episodes.extend(discover_episodes(sp))
    return episodes


def slice_eeg_window(
    session_dir: str,
    start_ms: float,
    end_ms: float,
    target_T: int,
) -> np.ndarray:
    """Return an (8, target_T) z-scored EEG window from session data.csv.

    Shorter actual windows are zero-padded after z-scoring; longer ones are
    truncated. Per-window per-channel z-score matches `train_lr.slice_windows`.
    """
    raw_cols = [f"raw_{c}" for c in CHANNELS]
    df = pd.read_csv(
        os.path.join(session_dir, "data.csv"),
        usecols=["relative_time_ms", *raw_cols],
    )
    times = df["relative_time_ms"].to_numpy()
    samples = df[raw_cols].to_numpy(dtype=np.float32)  # (N, 8)
    mask = (times >= start_ms) & (times < end_ms)
    window = samples[mask]
    if window.shape[0] == 0:
        return np.zeros((len(CHANNELS), target_T), dtype=np.float32)
    if window.shape[0] >= target_T:
        window = window[:target_T]
    else:
        pad = np.zeros((target_T - window.shape[0], window.shape[1]),
                       dtype=np.float32)
        window = np.concatenate([window, pad], axis=0)
    x = window.T  # (8, T)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def build_xy(
    episodes: List[HitlEpisode],
    target_T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize (X, y_block, y_bowl) for a list of episodes."""
    from training.clport.splits import color_to_idx

    X = np.zeros((len(episodes), len(CHANNELS), target_T), dtype=np.float32)
    y_block = np.zeros((len(episodes),), dtype=np.int64)
    y_bowl = np.zeros((len(episodes),), dtype=np.int64)
    for i, ep in enumerate(episodes):
        X[i] = slice_eeg_window(
            ep.eeg_dir, ep.goal_shown_ms, ep.place_done_ms, target_T)
        y_block[i] = color_to_idx[ep.block_color]
        y_bowl[i] = color_to_idx[ep.bowl_color]
    return X, y_block, y_bowl
