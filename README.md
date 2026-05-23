# Neurosity Workspace

EEG recording, training, and offline inference for motor-imagery tasks on a
Neurosity Crown. Two task pipelines are included:

- **LR** — binary LEFT vs. RIGHT motor imagery.
- **item_mux** — N-item pickup task (default `PEN` / `HIGHLIGHTER`) with item
  positions reshuffled every reset; the model jointly predicts the cued item
  and its starting slot.
- **LaBraM finetuning** — generalized harness around the
  [LaBraM](https://github.com/935963004/LaBraM) foundation model, configurable
  per device (channel layout, sample rate) and per task (any combination of
  classify / regress / spatial-grid / token-decoder heads).
- **clport (HITL CLIPort)** — human-in-the-loop episode collection on a
  4-color `put-block-in-bowl` task using a forked
  [CLIPort](https://github.com/cliport/cliport) sim. Trains a two-head
  EEGNet `P(block_color, bowl_color | EEG)` and a CLIPort policy
  `Pi(action | frame, goal_text [, EEG])`. See
  [`docs/clport.md`](docs/clport.md) for the full pipeline, or the
  "clport task" section below for a quickstart.

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
`training/eegnet_models/<name>/lr_eegnet.pt` plus a sidecar JSON describing the
preprocessing (channels, `active_s`, `T`, normalization), `train_params.json`,
metrics CSV, and loss/accuracy/confusion plots.

### 3. Inference

```bash
python inference/inference_lr.py \
    --model training/eegnet_models/lr_baseline/lr_eegnet.pt \
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
`training/eegnet_models/<name>/item_mux_eegnet.pt` plus a sidecar JSON containing
the canonical `items` list and `n_locations`. Plots include combined loss,
per-head accuracy, and per-head confusion matrices for train and val.

All sessions in one training run must share the same item set; a session's
item set is inferred from its `events.csv` reset rows.

### 3. Inference

```bash
python inference/inference_item_mux.py \
    --model training/eegnet_models/mux_baseline/item_mux_eegnet.pt \
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

## LaBraM finetuning

Generalized harness for finetuning the LaBraM foundation model on any
10-10–compliant device. Two YAML configs describe a run independently:

- **device config** (`training/labram/configs/<device>.yaml`) — channel list,
  source sample rate, target rate, bandpass / notch, CSV layout. Adding a new
  device is a YAML edit; no code change.
- **task config** (`training/labram/configs/task_*.yaml`) — data root, window
  timing, and one or more heads (classify / regress / grid / token).
  Multiple heads on one task train jointly with weighted-sum loss.

Channels are masked via LaBraM's learned channel embedding — Neurosity's 8
electrodes (CP3, C3, F5, PO3, PO4, F6, C4, CP4) are passed by name and the
transformer attends only to those. No zero-padding or interpolation.

### 0. Submodule + checkpoint

```bash
git submodule update --init third_party/labram
# Then download a LaBraM pretrained checkpoint into
#   third_party/labram/checkpoints/labram-base.pth
# (release link in https://github.com/935963004/LaBraM)
```

### 1. Build cache (optional — `train.py` does it on demand)

```bash
python -m training.labram.preprocess \
    --device_config training/labram/configs/neurosity_crown.yaml \
    --task_config   training/labram/configs/task_lr.yaml \
    --session_ids 2026-05-20_09-52-51 2026-05-20_09-55-42
```

Writes `data/labram_cache/<task>/<session>/{data.h5,manifest.json}`. Re-runs
skip sessions whose source mtime + preprocessing hash are unchanged.

### 2. Train

```bash
python -m training.labram.train \
    --name labram_lr \
    --device_config training/labram/configs/neurosity_crown.yaml \
    --task_config   training/labram/configs/task_lr.yaml \
    --session_ids 2026-05-20_09-52-51 2026-05-20_09-55-42 \
    --epochs 50 --batch_size 16 \
    --pretrained_ckpt third_party/labram/checkpoints/labram-base.pth
```

Saves `training/labram_models/<name>/{labram.pt,sidecar.json,train_params.json,metrics.csv,loss.png,per_head_loss.png}`.
The sidecar carries the device + task configs so evaluation is reproducible.

**Small-data tip.** With only a few hundred training trials, the default
full-finetune recipe overfits within ~2 epochs (train_loss drops, val_loss
climbs). For datasets under ~1k trials, prefer a frozen backbone (linear
probe) — pass `--freeze_backbone --head_lr 1e-3 --epochs 30`. On the first
10 LR sessions (433 trials) the frozen-backbone run reached val_loss 0.67
and stayed stable, vs full-finetune's val_loss 0.67 at epoch 2 climbing to
0.78 by epoch 50.

Multi-head example using the existing item_mux data:

```bash
python -m training.labram.train \
    --name labram_item_mux \
    --device_config training/labram/configs/neurosity_crown.yaml \
    --task_config   training/labram/configs/task_item_mux.yaml \
    --session_ids 2026-05-20_15-50-12 \
    --epochs 50 --batch_size 16
```

### 3. Evaluate

```bash
python -m training.labram.evaluate \
    --name labram_lr \
    --session_ids react_left_holdout react_right_holdout \
    --data_root data/lr
```

Reports per-head accuracy / MSE / token-accuracy as appropriate.

## clport task (HITL CLIPort)

Human-in-the-loop episode collection on a 4-color `put-block-in-bowl`
sim, with EEG aligned to the entire scene. Trains:

- `P(block_color, bowl_color | EEG)` — two-head EEGNet
  ([`training/train_clport_eegnet.py`](training/train_clport_eegnet.py)).
- `Pi(action | frame, goal_text [, EEG])` — CLIPort policy via
  [`training/train_clport_policy.py`](training/train_clport_policy.py).

CLIPort lives as a submodule under
[`third_party/cliport`](third_party/cliport) (forked at
`https://github.com/colbhben/cliport`). See
[`docs/clport.md`](docs/clport.md) for the full setup, recording flow,
training commands, split definitions, and current status.

```bash
# Quickstart (Crown attached, env set up per docs/clport.md).
python workspace/record_clport.py --session_length 20 --pair_pool train
python training/train_clport_eegnet.py \
    --name clport_eegnet_v0 \
    --session_dirs data/clport/<session_id> \
    --epochs 50
```
