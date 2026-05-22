"""CSV session -> LaBraM-shaped HDF5 cache.

For each cue in `events.csv`, slice the raw EEG, apply LaBraM-required
preprocessing (bandpass + notch + resample to target rate + reshape to
`(C, A, patch_size)`), and persist the trial tensor + per-head labels.

Cache layout::

    data/labram_cache/<task>/<session_id>/data.h5
        x:      float32  (N, C, A, P)         per-trial input tensor
        labels/<head_name>: int64 / float32   per-trial label array
        meta:   JSON in attrs                 hash, channels, dt, etc.
    data/labram_cache/<task>/<session_id>/manifest.json
        records source mtime + preprocessing hash for invalidation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos

from training._session_io import (
    SessionTiming,
    check_cross_session_timing,
    load_session_csv,
    slice_cue_window,
)
from training.labram.config import (
    DeviceConfig,
    HeadConfig,
    TaskConfig,
    load_device_config,
    load_task_config,
    preprocessing_hash,
)


PATCH_SIZE = 200  # LaBraM `labram_*_patch200_200` patch length (in samples).


def _design_filters(device: DeviceConfig):
    """Return (bandpass_sos, notch_sos_or_None) at the device's source rate."""
    fs = device.sample_rate_hz
    low, high = device.bandpass_hz
    nyq = fs / 2.0
    bp_b, bp_a = butter(4, [low / nyq, min(high, nyq - 1.0) / nyq], btype="band")
    bp_sos = tf2sos(bp_b, bp_a)
    notch_sos = None
    if device.notch_hz:
        nb, na = iirnotch(device.notch_hz / nyq, Q=30.0)
        notch_sos = tf2sos(nb, na)
    return bp_sos, notch_sos


def _impute_leading_nans(samples: np.ndarray) -> np.ndarray:
    """Replace any NaN/Inf in (T, C) by the first finite value per channel.

    Neurosity's rawUnfiltered stream lags the raw stream, so the logger emits
    a CSV with empty cells for ~1-2 s at the start. sosfiltfilt would explode
    those into all-NaN output. Forward-fill from the first finite sample
    (back-fills the leading gap) and replace any remaining NaN with 0 so
    filters stay stable.
    """
    x = np.asarray(samples, dtype=np.float64)
    for c in range(x.shape[1]):
        col = x[:, c]
        bad = ~np.isfinite(col)
        if not bad.any():
            continue
        good_idx = np.where(~bad)[0]
        if good_idx.size == 0:
            x[:, c] = 0.0
            continue
        first = good_idx[0]
        # Backfill the leading NaN run with the first finite value.
        col[:first] = col[first]
        # Forward-fill anything else.
        bad = ~np.isfinite(col)
        if bad.any():
            last_good = col[first]
            for i in range(first, len(col)):
                if np.isfinite(col[i]):
                    last_good = col[i]
                else:
                    col[i] = last_good
        x[:, c] = col
    return x


def _filter_resample(samples: np.ndarray, device: DeviceConfig) -> np.ndarray:
    """Apply bandpass + optional notch, then resample to target rate.

    Operates along axis 0 (time). Input shape: (T, C). Output: (T', C).
    """
    samples = _impute_leading_nans(samples)
    bp_sos, notch_sos = _design_filters(device)
    x = sosfiltfilt(bp_sos, samples, axis=0)
    if notch_sos is not None:
        x = sosfiltfilt(notch_sos, x, axis=0)
    src = device.sample_rate_hz
    dst = device.target_sample_rate_hz
    if src != dst:
        # resample_poly with up=dst, down=src; reduce by gcd for numerical stability.
        from math import gcd

        g = gcd(src, dst)
        x = resample_poly(x, up=dst // g, down=src // g, axis=0)
    return x.astype(np.float32, copy=False)


def _trial_window_seconds(task: TaskConfig, timing: SessionTiming) -> float:
    if task.window_seconds is not None:
        return float(task.window_seconds)
    end_ms = task.window_end_ms if task.window_end_ms is not None else timing.active_s * 1000.0
    return (end_ms - task.window_start_ms) / 1000.0


def _label_for_head(row: pd.Series, head: HeadConfig) -> Optional[object]:
    """Extract the label value for one head from a `cue` events.csv row.

    Returns None if the row lacks the field or the value is not in the head's
    label vocabulary (e.g. unrelated cue rows). Callers should drop such trials
    for that head only — multi-head training tolerates per-head missing labels.
    """
    if head.type == "classify":
        field = head.label_field or "label"
        if field not in row or pd.isna(row[field]):
            return None
        val = row[field]
        if head.classes is not None:
            try:
                return head.classes.index(str(val))
            except ValueError:
                return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None
    if head.type == "regress":
        fields = head.target_fields or []
        if not fields:
            return None
        vec = []
        for f in fields:
            if f not in row or pd.isna(row[f]):
                return None
            vec.append(float(row[f]))
        return np.asarray(vec, dtype=np.float32)
    if head.type == "grid":
        # Convention: events.csv carries `grid_row` and `grid_col` columns.
        if "grid_row" not in row or "grid_col" not in row:
            return None
        if pd.isna(row["grid_row"]) or pd.isna(row["grid_col"]):
            return None
        r = int(row["grid_row"])
        c = int(row["grid_col"])
        return r * head.cols + c
    if head.type == "token":
        field = head.text_field or "text"
        if field not in row or pd.isna(row[field]):
            return None
        text = str(row[field]).split()
        ids = []
        for tok in text[: (head.max_len - 1)]:
            if tok in head.vocab:
                ids.append(head.vocab.index(tok))
            else:
                ids.append(0)  # 0 reserved for <unk>
        ids.append(len(head.vocab))  # <eos>
        # Pad to max_len with -1 (ignore_index).
        out = np.full((head.max_len,), -1, dtype=np.int64)
        out[: len(ids)] = ids
        return out
    raise ValueError(f"Unsupported head type {head.type!r}")


def _preprocess_session(
    session_dir: str,
    device: DeviceConfig,
    task: TaskConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], int, SessionTiming]:
    """Preprocess one session. Returns (x, labels_dict, masks_dict, A, timing).

    `x` shape: (N, C, A, P). `labels_dict[head]` is a stacked array; entries
    where the trial had no label for that head are filled with the head's
    ignore value and `masks_dict[head]` is False there.
    """
    times, samples, timing, events = load_session_csv(
        session_dir,
        device.channels,
        time_col=device.csv.time_col,
        channel_col_template=device.csv.channel_col_template,
    )
    # Filter + resample once for the whole session, then slice in target time.
    samples_proc = _filter_resample(samples, device)
    src = device.sample_rate_hz
    dst = device.target_sample_rate_hz
    times_proc = np.arange(samples_proc.shape[0]) * (1000.0 / dst)
    # Keep cue alignment in real time: cue's relative_time_ms refers to wall
    # ms since session t=0, which both source and resampled streams share.

    window_s = _trial_window_seconds(task, timing)
    target_T = int(round(window_s * dst))
    if target_T % PATCH_SIZE != 0:
        # Trim to nearest lower multiple so reshape lands on whole patches.
        target_T = (target_T // PATCH_SIZE) * PATCH_SIZE
    if target_T < PATCH_SIZE:
        raise ValueError(
            f"window {window_s:.3f}s @ {dst}Hz is shorter than one {PATCH_SIZE}-sample patch"
        )
    A = target_T // PATCH_SIZE

    x_list: List[np.ndarray] = []
    label_lists: Dict[str, List[object]] = {h.name: [] for h in task.heads}
    mask_lists: Dict[str, List[bool]] = {h.name: [] for h in task.heads}

    cues = events[events["event"] == "cue"]
    for _, row in cues.iterrows():
        cue_ms = float(row["relative_time_ms"])
        # window_end_ms below is in source time but we slice from the
        # resampled stream — both share the same `times` (ms since t=0).
        win = slice_cue_window(
            times_proc,
            samples_proc,
            cue_ms,
            window_start_ms=task.window_start_ms,
            window_end_ms=task.window_start_ms + window_s * 1000.0,
            target_T=target_T,
            pad=True,
        )
        if win is None:
            continue
        # win: (T, C).
        win = win.astype(np.float32, copy=False)
        if device.normalization == "per_window_zscore":
            mean = win.mean(axis=0, keepdims=True)
            std = win.std(axis=0, keepdims=True) + 1e-6
            win = (win - mean) / std
        elif device.normalization == "uv":
            # No further rescaling — assume input is already in µV (Neurosity raw is).
            pass
        else:
            raise ValueError(f"Unknown normalization {device.normalization!r}")
        # Reshape to (C, A, P).
        x = win.T.reshape(win.shape[1], A, PATCH_SIZE)
        x_list.append(x)

        for head in task.heads:
            label = _label_for_head(row, head)
            if label is None:
                label_lists[head.name].append(None)
                mask_lists[head.name].append(False)
            else:
                label_lists[head.name].append(label)
                mask_lists[head.name].append(True)

    if not x_list:
        return (
            np.empty((0, len(device.channels), 0, PATCH_SIZE), dtype=np.float32),
            {h.name: np.empty((0,), dtype=np.int64) for h in task.heads},
            {h.name: np.empty((0,), dtype=bool) for h in task.heads},
            A,
            timing,
        )

    x_arr = np.stack(x_list)
    labels_arr: Dict[str, np.ndarray] = {}
    masks_arr: Dict[str, np.ndarray] = {}
    for head in task.heads:
        masks = np.asarray(mask_lists[head.name], dtype=bool)
        masks_arr[head.name] = masks
        if head.type in ("classify", "grid"):
            arr = np.zeros((len(x_list),), dtype=np.int64)
            for i, v in enumerate(label_lists[head.name]):
                if v is not None:
                    arr[i] = int(v)
            labels_arr[head.name] = arr
        elif head.type == "regress":
            arr = np.zeros((len(x_list), head.dim), dtype=np.float32)
            for i, v in enumerate(label_lists[head.name]):
                if v is not None:
                    arr[i] = v
            labels_arr[head.name] = arr
        elif head.type == "token":
            arr = np.full((len(x_list), head.max_len), -1, dtype=np.int64)
            for i, v in enumerate(label_lists[head.name]):
                if v is not None:
                    arr[i] = v
            labels_arr[head.name] = arr

    return x_arr, labels_arr, masks_arr, A, timing


def _cache_dir(cache_root: str, task: TaskConfig, device: DeviceConfig, session_id: str) -> str:
    """Cache path is keyed by (task_name, preprocessing_hash, session_id).

    Including the hash in the path means runs with different preprocessing
    (e.g. different `window_seconds`, channel set, sample rate) get separate
    caches and never overwrite each other.
    """
    h = preprocessing_hash(device, task)
    return os.path.join(cache_root, task.name, h, session_id)


def _source_mtime(session_dir: str) -> float:
    paths = [os.path.join(session_dir, "data.csv"), os.path.join(session_dir, "events.csv")]
    return max(os.path.getmtime(p) for p in paths)


def build_cache(
    session_ids: List[str],
    device: DeviceConfig,
    task: TaskConfig,
    *,
    cache_root: str = "data/labram_cache",
    force: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Preprocess each session into HDF5 (skip if up-to-date). Return cache dirs."""
    cfg_hash = preprocessing_hash(device, task)
    out_dirs: List[str] = []
    timings: List[SessionTiming] = []

    for sid in session_ids:
        session_dir = os.path.join(task.data_root, sid)
        cdir = _cache_dir(cache_root, task, device, sid)
        os.makedirs(cdir, exist_ok=True)
        manifest_path = os.path.join(cdir, "manifest.json")
        h5_path = os.path.join(cdir, "data.h5")

        src_mtime = _source_mtime(session_dir)
        manifest: Dict[str, object] = {}
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        is_fresh = (
            not force
            and manifest.get("preprocessing_hash") == cfg_hash
            and float(manifest.get("source_mtime", -1)) >= src_mtime
            and os.path.exists(h5_path)
        )
        if is_fresh:
            if verbose:
                with h5py.File(h5_path, "r") as f:
                    n = f["x"].shape[0]
                print(f"  [cache hit] {sid}: {n} trials")
            timings.append(
                SessionTiming(
                    active_s=float(manifest.get("active_s", 0.0)),
                    reset_s=float(manifest.get("reset_s", 0.0)),
                )
            )
            out_dirs.append(cdir)
            continue

        if verbose:
            t0 = time.time()
            print(f"  [build] {sid}: preprocessing...")
        x, labels, masks, A, timing = _preprocess_session(session_dir, device, task)
        timings.append(timing)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("x", data=x, compression="gzip", compression_opts=4)
            grp_l = f.create_group("labels")
            grp_m = f.create_group("masks")
            for name in labels:
                grp_l.create_dataset(name, data=labels[name])
                grp_m.create_dataset(name, data=masks[name])
            f.attrs["channels"] = json.dumps(list(device.channels))
            f.attrs["target_sample_rate_hz"] = device.target_sample_rate_hz
            f.attrs["patch_size"] = PATCH_SIZE
            f.attrs["A"] = A
            f.attrs["preprocessing_hash"] = cfg_hash
        manifest = {
            "preprocessing_hash": cfg_hash,
            "source_mtime": src_mtime,
            "n_trials": int(x.shape[0]),
            "channels": list(device.channels),
            "patch_size": PATCH_SIZE,
            "A": A,
            "target_sample_rate_hz": device.target_sample_rate_hz,
            "active_s": timing.active_s,
            "reset_s": timing.reset_s,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        if verbose:
            print(f"    -> {x.shape[0]} trials in {time.time() - t0:.1f}s")
        out_dirs.append(cdir)

    if len(timings) >= 2:
        check_cross_session_timing(session_ids, timings)
    return out_dirs


def main():
    parser = argparse.ArgumentParser(description="Preprocess sessions into LaBraM cache")
    parser.add_argument("--device_config", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--session_ids", nargs="+", required=True)
    parser.add_argument("--cache_root", default="data/labram_cache")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = load_device_config(args.device_config)
    task = load_task_config(args.task_config)
    build_cache(
        args.session_ids,
        device,
        task,
        cache_root=args.cache_root,
        force=args.force,
    )


if __name__ == "__main__":
    main()
