"""Shared CSV-session loading helpers.

Used by `training/train_lr.py`, `training/train_item_mux.py`, and the LaBraM
finetuning harness so they all parse `data.csv` / `events.csv` the same way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


TIMING_TOLERANCE_MS = 50.0


@dataclass
class SessionTiming:
    active_s: float
    reset_s: float


def infer_timing(events_df: pd.DataFrame) -> SessionTiming:
    """Infer mean active/reset durations from an events.csv DataFrame."""
    rows = events_df.to_dict("records")
    if len(rows) < 2:
        raise ValueError("events.csv has fewer than 2 rows; cannot infer timing")

    active_intervals = []
    reset_intervals = []
    for i, row in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if nxt is None:
            continue
        if row["event"] == "cue" and nxt["event"] == "reset":
            active_intervals.append(nxt["relative_time_ms"] - row["relative_time_ms"])
        if row["event"] == "reset" and nxt["event"] == "cue":
            reset_intervals.append(nxt["relative_time_ms"] - row["relative_time_ms"])

    if not active_intervals:
        raise ValueError("No cue->reset pairs found in events.csv")

    active_ms = float(np.mean(active_intervals))
    reset_ms = float(np.mean(reset_intervals)) if reset_intervals else 0.0
    return SessionTiming(active_s=active_ms / 1000.0, reset_s=reset_ms / 1000.0)


def load_session_csv(
    session_dir: str,
    channels: List[str],
    *,
    time_col: str = "relative_time_ms",
    channel_col_template: str = "raw_{ch}",
) -> Tuple[np.ndarray, np.ndarray, SessionTiming, pd.DataFrame]:
    """Load a session into (times_ms, samples (N,C), timing, events_df).

    Other event metadata columns are returned untouched in the events DataFrame.
    """
    data_path = os.path.join(session_dir, "data.csv")
    events_path = os.path.join(session_dir, "events.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    if not os.path.exists(events_path):
        raise FileNotFoundError(events_path)

    raw_cols = [channel_col_template.format(ch=c) for c in channels]
    data = pd.read_csv(data_path, usecols=[time_col, *raw_cols])
    events = pd.read_csv(events_path)
    timing = infer_timing(events)

    times = data[time_col].to_numpy()
    samples = data[raw_cols].to_numpy(dtype=np.float32)
    return times, samples, timing, events


def slice_cue_window(
    times: np.ndarray,
    samples: np.ndarray,
    cue_ms: float,
    window_start_ms: float,
    window_end_ms: float,
    target_T: int,
    *,
    pad: bool = True,
) -> Optional[np.ndarray]:
    """Cut one (target_T, C) window aligned to cue_ms. Returns None if empty.

    If `pad` is True (default), short windows are zero-padded at the tail to
    `target_T`. If False, returns None when fewer than `target_T` samples
    are available.
    """
    t_start = cue_ms + window_start_ms
    t_end = cue_ms + window_end_ms
    mask = (times >= t_start) & (times < t_end)
    window = samples[mask]
    if window.shape[0] == 0:
        return None
    if window.shape[0] >= target_T:
        return window[:target_T]
    if not pad:
        return None
    fill = np.zeros((target_T - window.shape[0], window.shape[1]), dtype=window.dtype)
    return np.concatenate([window, fill], axis=0)


def check_cross_session_timing(
    session_ids: List[str],
    timings: List[SessionTiming],
    *,
    tolerance_ms: float = TIMING_TOLERANCE_MS,
) -> None:
    base = timings[0]
    for sid, t in zip(session_ids, timings):
        if abs(t.active_s - base.active_s) * 1000 > tolerance_ms:
            raise ValueError(
                f"Session {sid} active_s={t.active_s:.3f} differs from "
                f"{session_ids[0]} active_s={base.active_s:.3f} "
                f"(tolerance {tolerance_ms}ms)"
            )
        if abs(t.reset_s - base.reset_s) * 1000 > tolerance_ms:
            raise ValueError(
                f"Session {sid} reset_s={t.reset_s:.3f} differs from "
                f"{session_ids[0]} reset_s={base.reset_s:.3f} "
                f"(tolerance {tolerance_ms}ms)"
            )
