"""Color-session -> DANCE batch dict adapter.

Mirrors `third_party/dance/dance/example/data.py` but sources data from this
repo's `data/color/<timestamp>/{data.csv,events.csv}` sessions via the same
loaders used by `training/train_color.py` and the LaBraM harness, so the per-
trial cue window is sliced bit-for-bit identically across all three trainers.

Each color trial becomes a single full-window event:

    eeg     : (C=8, T) float32 at 256Hz, per-window per-channel z-scored
    start   : 0.0       (window-relative, in [0, 1])
    end     : 1.0
    class   : label_to_id[label] + 1   (1..N; 0 reserved for padding)

Class 0 is DANCE's mandatory background/padding class. We use slot 0 of the
zero-padded `(MAX_EVENTS,)` event arrays to carry the one real event per
trial; remaining slots stay at class=0 so DANCE's loss stack treats them as
"no event" (matching the example collate_fn).

`channel_positions` is intentionally absent — `Dance(use_channel_merger=False)`
ignores it and the conv stack consumes our raw 8 channels directly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Allow `from training._session_io import ...` when this module is imported
# either as `training.dance.dataset` or directly via the trainer's sys.path
# bootstrap.
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from training._session_io import (  # noqa: E402
    SessionTiming,
    check_cross_session_timing,
    load_session_csv,
    slice_cue_window,
)

CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
SAMPLING_RATE = 256
TIMING_TOLERANCE_MS = 50.0


@dataclass(frozen=True)
class ColorTrial:
    """One cue-window's worth of EEG + its color label id."""

    eeg: np.ndarray  # (C, T) float32, z-scored
    class_id: int  # 1..N (DANCE convention; 0 is reserved for padding)


class ColorDanceDataset(Dataset):
    """In-memory color trials shaped for `Dance(use_channel_merger=False)`."""

    def __init__(self, trials: List[ColorTrial]) -> None:
        self.trials = trials

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> ColorTrial:
        return self.trials[idx]


def _slice_session(
    times: np.ndarray,
    samples: np.ndarray,
    events,
    target_T: int,
    window_start_ms: float,
    window_end_ms: float,
    label_to_id: Dict[str, int],
) -> List[ColorTrial]:
    out: List[ColorTrial] = []
    for _, row in events.iterrows():
        if row["event"] != "cue":
            continue
        label = row.get("label")
        if label not in label_to_id:
            continue
        cue_ms = float(row["relative_time_ms"])
        win = slice_cue_window(
            times, samples, cue_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            target_T=target_T,
            pad=True,
        )
        if win is None:
            continue
        x = win.T  # (C, T)
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-6
        x = ((x - mean) / std).astype(np.float32, copy=False)
        # +1 because DANCE reserves class 0 for "no event" / padding.
        out.append(ColorTrial(eeg=x, class_id=label_to_id[label] + 1))
    return out


def build_color_trials(
    session_ids: List[str],
    data_root: str,
    label_to_id: Dict[str, int],
    *,
    window_start_ms: float = 0.0,
    window_end_ms: float | None = None,
) -> Tuple[List[ColorTrial], SessionTiming, float]:
    """Slice color sessions into per-cue full-window trials for DANCE.

    Returns (trials, base_timing, window_seconds). `window_seconds` is the
    duration the trainer should hand to `Dance(duration=...)` so per-token
    rates land at the actual sliced-window length.
    """
    timings: List[SessionTiming] = []
    per_session = []
    for sid in session_ids:
        session_dir = os.path.join(data_root, sid)
        times, samples, timing, events = load_session_csv(session_dir, CHANNELS)
        timings.append(timing)
        per_session.append((sid, times, samples, events))

    check_cross_session_timing(session_ids, timings, tolerance_ms=TIMING_TOLERANCE_MS)
    base = timings[0]
    active_ms = base.active_s * 1000.0

    if window_end_ms is None:
        window_end_ms = active_ms
    if window_start_ms < 0:
        raise ValueError(f"window_start_ms must be >= 0 (got {window_start_ms})")
    if window_end_ms <= window_start_ms:
        raise ValueError(
            f"window_end_ms ({window_end_ms}) must be > window_start_ms ({window_start_ms})"
        )
    if window_end_ms > active_ms + TIMING_TOLERANCE_MS:
        raise ValueError(
            f"window_end_ms ({window_end_ms}) exceeds active period "
            f"({active_ms:.1f} ms) for session {session_ids[0]}"
        )

    target_T = int(round((window_end_ms - window_start_ms) / 1000.0 * SAMPLING_RATE))
    trials: List[ColorTrial] = []
    for sid, times, samples, events in per_session:
        chunk = _slice_session(
            times, samples, events, target_T,
            window_start_ms, window_end_ms, label_to_id,
        )
        print(f"  session {sid}: {len(chunk)} episodes")
        trials.extend(chunk)

    window_seconds = (window_end_ms - window_start_ms) / 1000.0
    return trials, base, window_seconds


def make_collate(n_queries: int):
    """Return a `collate_fn` that stacks `ColorTrial`s into a DANCE batch dict.

    Per-event arrays are zero-padded to `n_queries` (matching DANCE's
    convention that `max_events == n_queries`); slot 0 carries the one real
    full-window event, the rest stay at class=0 (= no event).
    """

    def collate(batch: List[ColorTrial]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        eeg = torch.from_numpy(np.stack([t.eeg for t in batch]))  # (B, C, T)
        starts = torch.zeros(B, n_queries, dtype=torch.float32)
        ends = torch.zeros(B, n_queries, dtype=torch.float32)
        classes = torch.zeros(B, n_queries, dtype=torch.long)
        for i, t in enumerate(batch):
            starts[i, 0] = 0.0
            ends[i, 0] = 1.0
            classes[i, 0] = int(t.class_id)
        return {
            "eeg": eeg,
            "start": starts,
            "end": ends,
            "class": classes,
        }

    return collate
