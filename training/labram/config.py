"""Config dataclasses + YAML loader for the LaBraM finetuning harness.

Two YAML files combine to describe a run:
- device config: hardware and CSV layout (channels, sample rate, columns).
- task config: data root, window timing, and head specifications.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class CsvLayout:
    time_col: str = "relative_time_ms"
    channel_col_template: str = "raw_{ch}"


@dataclass
class DeviceConfig:
    name: str
    channels: List[str]
    sample_rate_hz: int
    target_sample_rate_hz: int = 200
    units: str = "uV"
    notch_hz: Optional[float] = 50.0
    bandpass_hz: Tuple[float, float] = (0.1, 75.0)
    csv: CsvLayout = field(default_factory=CsvLayout)
    # "per_window_zscore": each trial's per-channel mean/std normalized to 0/1.
    #   Matches train_lr.py / train_item_mux.py and protects against scale drift
    #   between sessions, but discards µV scale that LaBraM was pretrained on.
    # "uv": pass through whatever scale the input is already in (µV).
    #   Closer to LaBraM's pretraining distribution; recommended when feeding
    #   rawUnfiltered + the LaBraM filter recipe.
    normalization: str = "per_window_zscore"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeviceConfig":
        d = dict(d)
        if "csv" in d and isinstance(d["csv"], dict):
            d["csv"] = CsvLayout(**d["csv"])
        if "bandpass_hz" in d:
            d["bandpass_hz"] = tuple(d["bandpass_hz"])
        return cls(**d)


@dataclass
class HeadConfig:
    name: str
    type: str  # classify | regress | grid | token
    weight: float = 1.0
    # classify
    classes: Optional[List[str]] = None
    num_classes: Optional[int] = None
    label_field: Optional[str] = None  # which key in events.csv (default: "label")
    # regress
    dim: Optional[int] = None
    target_fields: Optional[List[str]] = None
    loss: Optional[str] = None  # mse | huber
    # grid
    rows: Optional[int] = None
    cols: Optional[int] = None
    # token decoder
    vocab: Optional[List[str]] = None
    max_len: Optional[int] = None
    text_field: Optional[str] = None

    def __post_init__(self):
        if self.type == "classify":
            if self.classes is not None and self.num_classes is None:
                self.num_classes = len(self.classes)
            if self.num_classes is None:
                raise ValueError(f"head {self.name}: classify needs classes or num_classes")
        elif self.type == "regress":
            if self.dim is None:
                raise ValueError(f"head {self.name}: regress needs dim")
            if self.loss not in (None, "mse", "huber"):
                raise ValueError(f"head {self.name}: loss must be mse|huber")
            if self.loss is None:
                self.loss = "mse"
        elif self.type == "grid":
            if self.rows is None or self.cols is None:
                raise ValueError(f"head {self.name}: grid needs rows and cols")
        elif self.type == "token":
            if not self.vocab or not self.max_len:
                raise ValueError(f"head {self.name}: token needs vocab and max_len")
        else:
            raise ValueError(f"head {self.name}: unknown type {self.type!r}")


@dataclass
class TaskConfig:
    name: str
    data_root: str
    window_start_ms: float = 0.0
    window_end_ms: Optional[float] = None  # None = whole active period
    window_seconds: Optional[float] = None  # if set, overrides start/end
    heads: List[HeadConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskConfig":
        d = dict(d)
        heads_raw = d.pop("heads", {})
        heads: List[HeadConfig] = []
        if isinstance(heads_raw, dict):
            for name, spec in heads_raw.items():
                heads.append(HeadConfig(name=name, **spec))
        elif isinstance(heads_raw, list):
            for spec in heads_raw:
                heads.append(HeadConfig(**spec))
        d["heads"] = heads
        return cls(**d)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_device_config(path: str) -> DeviceConfig:
    return DeviceConfig.from_dict(load_yaml(path))


def load_task_config(path: str) -> TaskConfig:
    return TaskConfig.from_dict(load_yaml(path))


def preprocessing_hash(device: DeviceConfig, task: TaskConfig) -> str:
    """Stable digest of the preprocessing-relevant fields. Used for cache keys."""
    payload = {
        "device": {
            "channels": list(device.channels),
            "sample_rate_hz": device.sample_rate_hz,
            "target_sample_rate_hz": device.target_sample_rate_hz,
            "units": device.units,
            "notch_hz": device.notch_hz,
            "bandpass_hz": list(device.bandpass_hz),
            "csv": asdict(device.csv),
            "normalization": device.normalization,
        },
        "task": {
            "name": task.name,
            "window_start_ms": task.window_start_ms,
            "window_end_ms": task.window_end_ms,
            "window_seconds": task.window_seconds,
        },
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
