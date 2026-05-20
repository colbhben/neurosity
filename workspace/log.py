"""Streaming logger for the Neurosity SDK.

Records raw, rawUnfiltered, psd, powerByBand, and signal_quality streams to a
single CSV at 256Hz. The `raw` stream acts as the timing spine; values from the
slower streams are forward-filled (snapshotted at raw-epoch arrival).

Other modules use this by constructing `Logger(...)`, calling `start()` (which
blocks until the first raw sample establishes t=0), and `stop()` when done.
`relative_ms()` exposes the same t=0 reference so other recorders can stay
time-synchronized.
"""

import csv
import json
import os
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Optional

from neurosity import NeurositySDK


class Logger:
    SAMPLING_RATE = 256
    CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
    BANDS = ["alpha", "beta", "delta", "gamma", "theta"]

    def __init__(
        self,
        device_id: str,
        email: str,
        password: str,
        data_root: str = "data",
        session_dir: Optional[str] = None,
        environment: str = "production",
    ):
        self.device_id = device_id
        self.email = email
        self.password = password
        self.data_root = data_root
        self.session_dir = session_dir
        self.environment = environment

        self._sdk: Optional[NeurositySDK] = None
        self._unsubs = []

        self._lock = threading.Lock()
        self._latest_unfiltered = None
        self._latest_psd = None
        self._latest_power = None
        self._latest_signal = None

        self._queue: Queue = Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._csv_file = None
        self._csv_writer = None

        self._sample_count = 0
        self._enqueued_samples = 0
        self._t0_monotonic: Optional[float] = None
        self._first_sample_event = threading.Event()
        self._stop_sentinel = object()
        self._raw_listeners = []
        self._listener_lock = threading.Lock()

    def _ensure_session_dir(self) -> str:
        if self.session_dir is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir = os.path.join(self.data_root, ts)
        os.makedirs(self.session_dir, exist_ok=True)
        return self.session_dir

    def _build_columns(self):
        cols = ["relative_time_ms"]
        cols += [f"raw_{c}" for c in self.CHANNELS]
        cols += [f"rawUnfiltered_{c}" for c in self.CHANNELS]
        for b in self.BANDS:
            cols += [f"powerByBand_{b}_{c}" for c in self.CHANNELS]
        cols += ["psd_json", "signal_quality_json"]
        return cols

    def start(self) -> str:
        """Connect, subscribe, and block until the first raw sample arrives."""
        self._ensure_session_dir()
        csv_path = os.path.join(self.session_dir, "data.csv")
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self._build_columns())

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self._sdk = NeurositySDK(
            {"device_id": self.device_id, "environment": self.environment}
        )
        self._sdk.login({"email": self.email, "password": self.password})

        # Subscribe to slower / spine-supporting streams first so they have at
        # least one value cached by the time the first raw epoch lands.
        self._unsubs.append(self._sdk.brainwaves_raw_unfiltered(self._on_unfiltered))
        self._unsubs.append(self._sdk.brainwaves_psd(self._on_psd))
        self._unsubs.append(self._sdk.brainwaves_power_by_band(self._on_power))
        self._unsubs.append(self._sdk.signal_quality(self._on_signal))
        self._unsubs.append(self._sdk.brainwaves_raw(self._on_raw))

        self._first_sample_event.wait()
        return self.session_dir

    def stop(self):
        for u in self._unsubs:
            try:
                u()
            except Exception:
                pass
        self._unsubs = []
        self._queue.put(self._stop_sentinel)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5)
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def relative_ms(self) -> float:
        """Milliseconds since t=0 (first raw sample). 0 before logging starts."""
        if self._t0_monotonic is None:
            return 0.0
        return (time.monotonic() - self._t0_monotonic) * 1000.0

    def add_raw_listener(self, callback):
        """Register a callback fired for each raw epoch.

        Callback signature: callback(epoch, start_ms) where `epoch` is the
        list-of-channels payload from the SDK and `start_ms` is the
        relative_time_ms of the epoch's first sample (same clock as
        `data.csv`'s `relative_time_ms` column).
        """
        with self._listener_lock:
            self._raw_listeners.append(callback)

    def _on_unfiltered(self, data):
        with self._lock:
            self._latest_unfiltered = data.get("data")

    def _on_psd(self, data):
        with self._lock:
            self._latest_psd = data

    def _on_power(self, data):
        with self._lock:
            self._latest_power = data.get("data")

    def _on_signal(self, data):
        with self._lock:
            self._latest_signal = data

    def _on_raw(self, data):
        epoch = data.get("data")
        if not epoch or not epoch[0]:
            return
        if self._t0_monotonic is None:
            self._t0_monotonic = time.monotonic()
            self._first_sample_event.set()
        # Sample-counter timestamps drift if the SDK delivers epochs slower
        # than 256Hz on average; listeners need wall-clock ms so they share a
        # clock with relative_ms() / cue timestamps.
        epoch_arrival_ms = (time.monotonic() - self._t0_monotonic) * 1000.0
        n = len(epoch[0])
        start_ms = epoch_arrival_ms - (n - 1) * 1000.0 / self.SAMPLING_RATE
        self._enqueued_samples += n
        with self._lock:
            snapshot = (
                self._latest_unfiltered,
                self._latest_psd,
                self._latest_power,
                self._latest_signal,
            )
        self._queue.put((epoch, snapshot))
        with self._listener_lock:
            listeners = list(self._raw_listeners)
        for cb in listeners:
            try:
                cb(epoch, start_ms)
            except Exception:
                pass

    def _writer_loop(self):
        n_channels = len(self.CHANNELS)
        while True:
            item = self._queue.get()
            if item is self._stop_sentinel:
                return
            epoch, (unfiltered, psd, power, signal) = item
            samples_per_channel = len(epoch[0])
            psd_str = json.dumps(psd) if psd is not None else ""
            signal_str = json.dumps(signal) if signal is not None else ""
            for i in range(samples_per_channel):
                rel_ms = self._sample_count * 1000.0 / self.SAMPLING_RATE
                self._sample_count += 1
                row = [f"{rel_ms:.4f}"]
                for ch in range(n_channels):
                    row.append(
                        epoch[ch][i] if ch < len(epoch) and i < len(epoch[ch]) else ""
                    )
                for ch in range(n_channels):
                    if (
                        unfiltered is not None
                        and ch < len(unfiltered)
                        and i < len(unfiltered[ch])
                    ):
                        row.append(unfiltered[ch][i])
                    else:
                        row.append("")
                for b in self.BANDS:
                    for ch in range(n_channels):
                        if power is not None and b in power and ch < len(power[b]):
                            row.append(power[b][ch])
                        else:
                            row.append("")
                row.append(psd_str)
                row.append(signal_str)
                self._csv_writer.writerow(row)
            self._csv_file.flush()
