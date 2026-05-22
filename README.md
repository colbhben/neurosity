# Neurosity Workspace

EEG recording, training, and offline inference for motor-imagery tasks on a
Neurosity Crown. Two task pipelines are included:

- **LR** — binary LEFT vs. RIGHT motor imagery.
- **item_mux** — N-item pickup task (default `PEN` / `HIGHLIGHTER`) with item
  positions reshuffled every reset; the model jointly predicts the cued item
  and its starting slot.

## Setup

```bash
pip install -r requirements.txt -r dev-requirements.txt
pip install -e .
```

Create a `.env` in the repo root with:

```bash
NEUROSITY_DEVICE_ID=<32-char device id>
NEUROSITY_EMAIL=<your account email>
NEUROSITY_PASSWORD=<your account password>
```

Run all commands from the repo root.

## `workspace/log.py`

`Logger` is the streaming recorder. It subscribes to `raw`, `rawUnfiltered`,
`psd`, `powerByBand`, and `signal_quality`, and writes one CSV row per raw
sample (256Hz) to `<data_root>/<YYYY-MM-DD_HH-MM-SS>/data.csv`. The `raw`
stream is the timing spine; slower streams are forward-filled to each
raw-sample row. `start()` blocks until the first raw sample establishes
`t=0`, so other recorders can call `relative_ms()` and share the clock.

```python
from log import Logger

logger = Logger(
    device_id="...", email="...", password="...",
    data_root="data/lr",          # session is created under this dir
)
session_dir = logger.start()      # blocks until first raw sample (t=0)
print(logger.relative_ms())       # ms since t=0
# ... do stuff, e.g. write your own events.csv using relative_ms() ...
logger.stop()
```

`add_raw_listener(cb)` registers `cb(epoch, start_ms)` callbacks for live
consumers (each epoch's first sample is `start_ms` on the same clock as
`data.csv`'s `relative_time_ms`).

## LR task (LEFT / RIGHT)

### 1. Record

```bash
python workspace/record_lr.py \
    --episode_length 6 --reset_length 2 --session_length 20
```

Writes to `data/lr/<timestamp>/{data.csv,events.csv}`. `events.csv` rows are
`episode,event,label,relative_time_ms` where `event` is `cue` (with `label` =
`LEFT`/`RIGHT`) or `reset`.

### 2. Train

```bash
python training/train_lr.py \
    --name lr_baseline \
    --session_ids 2026-05-19_17-14-52 2026-05-20_09-12-03 \
    --epochs 50 --batch_size 16
```

Reads sessions from `data/lr/`. Saves
`training/models/<name>/lr_eegnet.pt` plus a sidecar JSON describing the
preprocessing (channels, `active_s`, `T`, normalization), `train_params.json`,
metrics CSV, and loss/accuracy/confusion plots.

### 3. Inference

```bash
python inference/inference_lr.py \
    --model training/models/lr_baseline/lr_eegnet.pt \
    --session_ids 2026-05-19_17-14-52 \
    --name lr_eval
```

Reads sessions from `data/lr/`. Writes `predictions.csv`, `accuracy.json`,
and figures (confusion matrix, per-episode timeline, per-session accuracy
bars, probability distribution) to `inference/data/<name>/`.

## item_mux task (N-item pickup)

Each reset randomly assigns each item to a unique slot `0..N-1`. The cue
picks one item at random; the user mentally "picks up" the item from its
known slot. The `events.csv` `location` column records the cued item's slot
at cue time, and the `locations` column records the slot layout (in slot
order, `;`-separated) at each reset. The trained model has two heads:
**item** (which item was cued) and **location** (which slot it was in).

### 1. Record

```bash
# Default items: PEN, HIGHLIGHTER.
python workspace/record_item_mux.py \
    --episode_length 6 --reset_length 2 --session_length 20

# Custom item set (>=2 unique names).
python workspace/record_item_mux.py \
    --episode_length 6 --reset_length 2 --session_length 20 \
    --items PEN HIGHLIGHTER ERASER
```

Writes to `data/item_mux/<timestamp>/{data.csv,events.csv}`. Each reset
prints `reset [0: <itemA> | 1: <itemB> | ...]`.

### 2. Train

```bash
python training/train_item_mux.py \
    --name mux_baseline \
    --session_ids 2026-05-19_17-14-52 \
    --epochs 50 --batch_size 16
```

Reads sessions from `data/item_mux/`. Saves
`training/models/<name>/item_mux_eegnet.pt` plus a sidecar JSON containing
the canonical `items` list and `n_locations`. Plots include combined loss,
per-head accuracy, and per-head confusion matrices for train and val.

All sessions in one training run must share the same item set; a session's
item set is inferred from its `events.csv` reset rows.

### 3. Inference

```bash
python inference/inference_item_mux.py \
    --model training/models/mux_baseline/item_mux_eegnet.pt \
    --session_ids 2026-05-19_17-14-52 \
    --name mux_eval
```

Reads sessions from `data/item_mux/`. Writes `predictions.csv`,
`accuracy.json`, and separate figures per head:

- `confusion_item.png`, `confusion_location.png`
- `timeline_item.png`, `timeline_location.png`
- `probability_distribution_item.png`, `probability_distribution_location.png`
- `per_session_accuracy.png` (item / location / joint bars)

Output goes to `inference/data/<name>/`.
