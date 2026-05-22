"""Dataset wrappers around the HDF5 cache produced by `preprocess.build_cache`."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from training.labram.config import DeviceConfig, TaskConfig


@dataclass
class Batch:
    x: torch.Tensor                      # (B, C, A, P)
    labels: Dict[str, torch.Tensor]      # name -> (B,) or (B, dim) or (B, L)
    masks: Dict[str, torch.Tensor]       # name -> (B,) bool


class LabramDataset(Dataset):
    """Concat dataset over per-session HDF5 caches.

    Each item is `(x, labels_dict, masks_dict)` where `x` is `(C, A, P)` and
    label dicts are keyed by head name. The corresponding `masks_dict` entry
    is False when this trial has no label for that head (skip in the loss).
    """

    def __init__(self, cache_dirs: Sequence[str], head_names: Sequence[str]):
        self.cache_dirs = list(cache_dirs)
        self.head_names = list(head_names)
        self._files: List[h5py.File] = []
        self._offsets: List[int] = []
        total = 0
        for cdir in self.cache_dirs:
            f = h5py.File(os.path.join(cdir, "data.h5"), "r")
            self._files.append(f)
            self._offsets.append(total)
            total += f["x"].shape[0]
        self._total = total
        # Sanity: shapes consistent across files.
        if self._files:
            ref = self._files[0]["x"].shape[1:]
            for f in self._files[1:]:
                if f["x"].shape[1:] != ref:
                    raise ValueError(
                        f"Cache shape mismatch: {f.filename} {f['x'].shape} vs ref {ref}"
                    )

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int):
        # Linear scan over files (cheap; usually <10 files per run).
        for i, off in enumerate(self._offsets):
            n = self._files[i]["x"].shape[0]
            if idx < off + n:
                return i, idx - off
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        fi, local = self._locate(idx)
        f = self._files[fi]
        x = torch.from_numpy(f["x"][local].astype(np.float32))
        labels: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        for name in self.head_names:
            arr = f["labels"][name][local]
            if np.issubdtype(arr.dtype, np.integer):
                labels[name] = torch.as_tensor(arr, dtype=torch.long)
            else:
                labels[name] = torch.as_tensor(arr, dtype=torch.float32)
            masks[name] = torch.as_tensor(bool(f["masks"][name][local]))
        return x, labels, masks

    def close(self):
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass

    @property
    def shape(self):
        if not self._files:
            return (0,)
        c, a, p = self._files[0]["x"].shape[1:]
        return (self._total, c, a, p)


def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    head_names = list(batch[0][1].keys())
    labels: Dict[str, torch.Tensor] = {}
    masks: Dict[str, torch.Tensor] = {}
    for name in head_names:
        labels[name] = torch.stack([b[1][name] for b in batch], dim=0)
        masks[name] = torch.stack([b[2][name] for b in batch], dim=0)
    return Batch(x=xs, labels=labels, masks=masks)


def read_cache_meta(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        return json.load(f)
