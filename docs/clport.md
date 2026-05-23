# CLIPort × Neurosity (HITL pipeline)

Human-in-the-loop data collection and training for two models on the
4-color `put-block-in-bowl` task:

- **`P(block_color, bowl_color | EEG)`** — two-head EEGNet over the
  goal-shown → place-done window of each episode.
- **`Pi(action | frame, goal_text [, EEG])`** — CLIPort policy with an
  optional EEG embedding fused into the language token.

Color pool: `blue, red, green, yellow`. 12 ordered (block, bowl) tuples;
4 held out diagonally, 8 used for training.

## 1. Install

CLIPort lives as a submodule under [`third_party/cliport/`](../third_party/cliport).
It pins `python==3.8`, `torch==1.7.1`, and `pybullet==3.0.4`, so it gets
its own env (kept on a target machine, not the dev laptop).

```bash
# Clone with submodules.
git clone --recurse-submodules <your-fork-of-this-repo> neurosity
cd neurosity
# If you cloned without --recurse-submodules:
git submodule update --init --recursive

# Pin Python via mise (or pyenv), then create the env. mise recipe:
mise use python@3.8
python3.8 -m venv .venv-clport
source .venv-clport/bin/activate

# Install neurosity + cliport in the same env so workspace/record_clport.py
# can import both.
pip install -r requirements.txt
pip install -e .
pip install -r third_party/cliport/requirements.txt
(cd third_party/cliport && python setup.py develop)
export CLIPORT_ROOT="$(pwd)/third_party/cliport"
```

Smoke-test the upstream pipeline (no Crown needed):

```bash
python third_party/cliport/cliport/demos.py \
    n=1 task=put-block-in-bowl-seen-colors mode=train disp=True
```

## 2. Record HITL episodes

Wear the Crown. The recorder runs the EEG `Logger` and a PyBullet GUI in
the same process and aligns clicks/frames/EEG to a shared `t=0`.

```bash
# 20 episodes from the 8 training tuples.
python workspace/record_clport.py --session_length 20 --pair_pool train

# 8 episodes from the 4 held-out tuples (val_unseen).
python workspace/record_clport.py --session_length 8 --pair_pool val_unseen

# Sim-only smoke test (no Crown).
python workspace/record_clport.py --session_length 2 --pair_pool train --no_eeg
```

Each session writes:

```
data/clport/<YYYY-MM-DD_HH-MM-SS>/
  data.csv                            # 256 Hz EEG (Logger format)
  events.csv                          # episode,event,label,relative_time_ms,...
  manifest.json
  episodes/<idx>-<seed>/
    obs.pkl   action.pkl   info.pkl   # CLIPort obs/action/info per step
    meta.json                         # block_color, bowl_color, success, ids, ...
    frames.mp4                        # full agent-cam video
    frame_index.csv                   # frame_idx <-> relative_time_ms
```

The HITL flow per episode:

1. Reset task — spawns one block + one bowl per HITL color (target +
   3 distractors so the human cannot guess by elimination).
2. PyBullet GUI shows the goal text overlay (`put the X block in the Y
   bowl`).
3. User clicks the correct-color block. Wrong clicks are rejected (not
   logged). Once correct, the oracle picks that block.
4. User clicks the correct-color bowl. Same wrong-click handling.
5. Oracle places the block.
6. Frame video + obs/action/info pickles are saved.

Events emitted: `goal_shown, click_block, click_bowl, pick_done,
place_done, episode_done`.

## 3. Train `P(goal_text | EEG)`

```bash
python training/train_clport_eegnet.py \
    --name clport_eegnet_v0 \
    --session_dirs data/clport/2026-05-22_10-00-00 data/clport/2026-05-22_11-30-00 \
    --window_seconds 8.0 \
    --epochs 50 --batch_size 16
```

- Slices `goal_shown → place_done` per episode, pads/truncates to
  `window_seconds × 256` samples.
- 4-class block head + 4-class bowl head share an EEGNet trunk (same
  shape as `training/train_lr.py:EEGNet`, so weights drop straight into
  the policy's EEG encoder).
- Reports per-split (val_seen / val_unseen / val_mixed) per-head and
  joint accuracy each epoch.

Outputs: `training/eegnet_models/clport_eegnet_v0/{clport_eegnet.pt,
clport_eegnet.json, metrics.csv, loss.png, joint_acc.png,
train_params.json}`.

Inference:

```bash
python inference/inference_clport_eegnet.py \
    --model_dir training/eegnet_models/clport_eegnet_v0 \
    --session_dirs data/clport/2026-05-22_* \
    --out_dir inference_runs/clport_eegnet_v0
```

Writes `predictions.csv`, `accuracy.json`, and per-split block/bowl
confusion matrices.

## 4. Train the policy

CLIPort's training loop expects RavensDataset-format pickles. The
wrapper materializes them, then calls upstream Hydra train.

```bash
# 1) Convert HITL sessions -> Ravens format. Done automatically by the
#    wrapper, but you can run it standalone:
python training/clport/build_ravens_dataset.py \
    --session_dirs data/clport/2026-05-22_* \
    --out_root data/clport_ravens \
    --success_only

# 2) Baseline (no EEG).
python training/train_clport_policy.py \
    --session_dirs data/clport/2026-05-22_* \
    --agent cliport \
    --n_demos 60 --n_steps 10000 --n_val 10 --success_only

# 3) EEG-conditioned (skeleton; see "Status" below).
python training/train_clport_policy.py \
    --session_dirs data/clport/2026-05-22_* \
    --agent two_stream_clip_lingunet_lat_transporter_eeg \
    --with_eeg --eeg_window_seconds 8.0 \
    --n_demos 60 --n_steps 10000 --n_val 10 --success_only
```

Eval:

```bash
python inference/inference_clport_policy.py \
    --ravens_root data/clport_ravens \
    --model_path training/clport_policy_runs/checkpoints \
    --train_config third_party/cliport/cliport/cfg/train.yaml \
    --agent cliport \
    --n_demos 20 \
    --out_dir inference_runs/clport_policy_baseline
```

## 5. Splits

`training/clport/splits.py` defines the 4-color splits:

| pool          | size | tuples |
|---------------|------|--------|
| `TRAIN_PAIRS` | 8    | every (block, bowl) with block ≠ bowl, minus the 4 held out |
| `HOLDOUT_UNSEEN` | 4 | `(red,blue), (blue,green), (green,yellow), (yellow,red)` |

`make_splits(episodes, seed)` returns:
- `train` — episodes from `TRAIN_PAIRS` (80%)
- `val_seen` — episodes from `TRAIN_PAIRS` (20%, every pair represented)
- `val_unseen` — every episode from `HOLDOUT_UNSEEN`
- `val_mixed` — balanced 50/50 mix of `val_seen` ∪ `val_unseen`

Both the EEGNet trainer and the Ravens-format converter use this single
splitter so they agree on which episodes are in which split.

## 6. Status

What works today:
- HITL task variant + click picker + recorder.
- Two-head EEGNet trainer + per-split reporter + offline inference.
- HITL-session-to-Ravens-format converter.
- Baseline `cliport` policy training/eval via the wrapper.

In progress:
- The EEG-conditioned agent
  ([`third_party/cliport/cliport/agents/eeg_conditioned.py`](../third_party/cliport/cliport/agents/eeg_conditioned.py))
  ships the `EEGEncoder` and registers
  `two_stream_clip_lingunet_lat_transporter_eeg`, but the EEG embedding
  is not yet fused into upstream `TwoStreamAttentionLangFusion` /
  `TwoStreamTransportLangFusion`. The class falls through to baseline
  CLIPort behavior until that integration is wired (see the module
  docstring for the two viable patterns).
- `RavensDataset.__getitem__` does not yet emit the `eeg.pkl` sidecars.
  An `EEGRavensDataset` subclass is the cleanest extension point.
