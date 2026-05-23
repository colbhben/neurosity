"""Human-in-the-loop CLIPort recorder for `put-block-in-bowl-hitl`.

For each episode:
  1. Reset the CLIPort env; spawn one block + one bowl per HITL color.
  2. Display the language goal in the PyBullet GUI.
  3. The user clicks the correct-color block -> oracle picks it.
  4. The user clicks the correct-color bowl   -> oracle places into it.
  5. Save (RGB-D obs at pick & place steps, pick/place actions, info,
     full-episode video, EEG-aligned frame index) to disk.

EEG comes from `workspace.log.Logger`, started once per session. Its first
raw sample defines `t = 0`; all event timestamps are recorded in
`relative_time_ms` against that clock so EEG <-> clicks <-> video frames
are alignable downstream.

Example calls (run from the repo root, with NEUROSITY_* set in .env):

    # 20-episode session over the training pair pool, EEG enabled.
    python workspace/record_clport.py --session_length 20 --pair_pool train

    # Held-out (val_unseen) compositions.
    python workspace/record_clport.py --session_length 10 --pair_pool val_unseen

    # Smoke test without the Crown (skips the EEG logger).
    python workspace/record_clport.py --session_length 2 --pair_pool train --no_eeg
"""

import argparse
import csv
import os
import pickle
import sys
import threading
import time
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

# Ensure both `workspace/` and the cliport submodule are importable when run
# from the repo root via `python workspace/record_clport.py`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))   # for `log`
sys.path.insert(0, os.path.join(_REPO_ROOT, "third_party", "cliport"))

from log import Logger  # noqa: E402

import pybullet as p  # noqa: E402

from cliport import tasks  # noqa: E402
from cliport.environments.environment import Environment  # noqa: E402
from cliport.environments.click_picker import ClickPicker  # noqa: E402
from cliport.tasks.put_block_in_bowl_hitl import HITL_COLORS  # noqa: E402


# Composition splits -- duplicated here so this script does not need to
# import torch via training/clport/splits.py. Kept in sync via tests.
_HOLDOUT_UNSEEN = [
    ("red", "blue"),
    ("blue", "green"),
    ("green", "yellow"),
    ("yellow", "red"),
]
_ALL_PAIRS = [(b, w) for b in HITL_COLORS for w in HITL_COLORS if b != w]
_TRAIN_PAIRS = [pr for pr in _ALL_PAIRS if pr not in _HOLDOUT_UNSEEN]


def pair_pool(name: str):
    if name == "train":
        return list(_TRAIN_PAIRS)
    if name == "val_unseen":
        return list(_HOLDOUT_UNSEEN)
    if name == "all":
        return list(_ALL_PAIRS)
    raise ValueError(f"unknown pair_pool: {name!r}")


def _run_video_recorder(env, video_path: str):
    """Wire env's built-in agent-camera video recorder to a fixed path."""
    record_cfg = {
        "save_video_path": os.path.dirname(video_path),
        "fps": 20,
        "video_height": 480,
        "video_width": 640,
        "add_text": False,
        "add_task_text": False,
    }
    env.record_cfg = record_cfg
    # filename is passed without extension; env appends .mp4
    stem = os.path.splitext(os.path.basename(video_path))[0]
    env.start_rec(stem)


def _frame_indexer_thread(stop_event, logger, frames_csv_path, hz=20):
    """Write (frame_idx, relative_time_ms) at ~`hz` so video frames can be
    aligned to data.csv. Frame indices are emitted in lock-step with the env
    saving a frame every `step_simulation` cycle (close to but not exactly
    `hz`); the index here is best-effort and intended for offline alignment
    of the entire episode, not per-frame sync.
    """
    f = open(frames_csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["wall_idx", "relative_time_ms"])
    i = 0
    dt = 1.0 / hz
    try:
        while not stop_event.is_set():
            w.writerow([i, f"{logger.relative_ms():.4f}"])
            f.flush()
            i += 1
            stop_event.wait(dt)
    finally:
        f.close()


def _events_writer(events_path: str):
    file = open(events_path, "w", newline="")
    w = csv.writer(file)
    w.writerow([
        "episode", "event", "label", "relative_time_ms",
        "block_color", "bowl_color", "clicked_obj_id", "attempts", "success",
        "seed", "pair_pool",
    ])
    return file, w


def _save_pickle(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _wait_for_settle(env, max_wait_s: float = 5.0):
    """Step the sim until objects are static or timeout."""
    t0 = time.monotonic()
    while not env.is_static and (time.monotonic() - t0) < max_wait_s:
        env.step_simulation()


def run_session(args):
    if args.assets_root is None:
        # Default to the assets bundled in the cliport submodule.
        args.assets_root = os.path.join(
            _REPO_ROOT, "third_party", "cliport", "cliport", "environments", "assets"
        )
    if not os.path.isdir(args.assets_root):
        raise FileNotFoundError(
            f"assets_root not found: {args.assets_root}. "
            "Pass --assets_root or run setup.py develop in third_party/cliport."
        )

    pool = pair_pool(args.pair_pool)
    if args.seed is not None:
        np.random.seed(args.seed)
        import random as _rng
        _rng.seed(args.seed)

    # Session directory (shared by EEG csv + per-episode dirs).
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(args.data_root, ts)
    episodes_root = os.path.join(session_dir, "episodes")
    os.makedirs(episodes_root, exist_ok=True)

    # Start EEG logger (or fake clock if --no_eeg).
    if args.no_eeg:
        print("[clport] EEG disabled (--no_eeg). Using monotonic clock.")
        os.makedirs(session_dir, exist_ok=True)
        t0 = time.monotonic()
        logger = type("FakeLogger", (), {
            "relative_ms": staticmethod(lambda: (time.monotonic() - t0) * 1000.0),
            "stop": staticmethod(lambda: None),
        })()
        used_session_dir = session_dir
    else:
        logger = Logger(
            device_id=args.device_id, email=args.email, password=args.password,
            data_root=args.data_root, session_dir=session_dir,
        )
        print("[clport] Connecting to Crown and waiting for first sample...")
        used_session_dir = logger.start()
        print(f"[clport] EEG t=0 set. Session dir: {used_session_dir}")

    events_path = os.path.join(used_session_dir, "events.csv")
    events_file, events = _events_writer(events_path)

    # Manifest captures the static config so downstream tools can
    # re-discover the session without re-parsing every csv.
    manifest_path = os.path.join(used_session_dir, "manifest.json")
    import json
    manifest = {
        "task": "put-block-in-bowl-hitl",
        "pair_pool": args.pair_pool,
        "session_length": args.session_length,
        "hitl_colors": list(HITL_COLORS),
        "no_eeg": args.no_eeg,
        "started_at": ts,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Build env (GUI required for human clicking).
    env = Environment(
        args.assets_root,
        disp=True,
        shared_memory=False,
        hz=480,
        record_cfg=None,
    )
    task_cls = tasks.names["put-block-in-bowl-hitl"]
    task = task_cls()
    task.mode = "train"
    task.allowed_pairs = pool
    env.set_task(task)

    successes = 0

    try:
        for ep in range(args.session_length):
            seed = (args.seed if args.seed is not None else 0) + ep * 2
            np.random.seed(seed)
            import random as _rng
            _rng.seed(seed)

            ep_dir = os.path.join(episodes_root, f"{ep:06d}-{seed}")
            os.makedirs(ep_dir, exist_ok=True)
            print(f"\n[clport] === Episode {ep + 1}/{args.session_length} (seed={seed}) ===")

            # --- Reset task & env ---
            obs = env.reset()
            info = dict(env.info)
            lang_goal = info["lang_goal"]
            block_color = task.block_color
            bowl_color = task.bowl_color
            print(f"[clport] goal: {lang_goal!r} (block={block_color}, bowl={bowl_color})")

            # --- Start video + frame index ---
            video_path = os.path.join(ep_dir, "frames.mp4")
            _run_video_recorder(env, video_path)
            stop_event = threading.Event()
            frame_csv = os.path.join(ep_dir, "frame_index.csv")
            t_index = threading.Thread(
                target=_frame_indexer_thread,
                args=(stop_event, logger, frame_csv, 20),
                daemon=True,
            )
            t_index.start()

            # --- goal_shown event ---
            events.writerow([
                ep, "goal_shown", lang_goal, f"{logger.relative_ms():.4f}",
                block_color, bowl_color, "", 0, "", seed, args.pair_pool,
            ])
            events_file.flush()

            # --- Click 1: block ---
            block_picker = ClickPicker(allowed_ids=task.block_ids)
            clicked_block_id, block_attempts, block_click_ms = block_picker.wait_for_click(
                prompt=f"click the {block_color} BLOCK",
                correct_id=task.target_block_id,
                clock_fn=logger.relative_ms,
            )
            events.writerow([
                ep, "click_block", block_color, f"{block_click_ms:.4f}",
                block_color, bowl_color, clicked_block_id, block_attempts, "",
                seed, args.pair_pool,
            ])
            events_file.flush()

            # --- Click 2: bowl ---
            bowl_picker = ClickPicker(allowed_ids=task.bowl_ids)
            clicked_bowl_id, bowl_attempts, bowl_click_ms = bowl_picker.wait_for_click(
                prompt=f"click the {bowl_color} BOWL",
                correct_id=task.target_bowl_id,
                clock_fn=logger.relative_ms,
            )
            events.writerow([
                ep, "click_bowl", bowl_color, f"{bowl_click_ms:.4f}",
                block_color, bowl_color, clicked_bowl_id, bowl_attempts, "",
                seed, args.pair_pool,
            ])
            events_file.flush()

            # --- Drive the oracle for the clicked pair ---
            task.constrain_goal_to_click(clicked_block_id, clicked_bowl_id)
            agent = task.oracle(env)

            episode_log = []
            total_reward = 0.0
            for step in range(task.max_steps):
                act = agent.act(obs, info)
                episode_log.append((obs, act, total_reward, info))
                obs, reward, done, info = env.step(act)
                total_reward += reward
                ev_name = "pick_done" if step == 0 else "place_done"
                events.writerow([
                    ep, ev_name, "", f"{logger.relative_ms():.4f}",
                    block_color, bowl_color, "", 0, "", seed, args.pair_pool,
                ])
                events_file.flush()
                if done:
                    break
            episode_log.append((obs, None, total_reward, info))

            # --- Stop video + frame index ---
            stop_event.set()
            t_index.join(timeout=2)
            env.end_rec()

            success = bool(total_reward > 0.99)
            if success:
                successes += 1

            # --- Persist per-episode artifacts ---
            obs_list = [step_tuple[0] for step_tuple in episode_log]
            act_list = [step_tuple[1] for step_tuple in episode_log]
            info_list = [step_tuple[3] for step_tuple in episode_log]
            _save_pickle(os.path.join(ep_dir, "obs.pkl"), obs_list)
            _save_pickle(os.path.join(ep_dir, "action.pkl"), act_list)
            _save_pickle(os.path.join(ep_dir, "info.pkl"), info_list)
            with open(os.path.join(ep_dir, "meta.json"), "w") as f:
                json.dump({
                    "episode": ep,
                    "seed": seed,
                    "pair_pool": args.pair_pool,
                    "block_color": block_color,
                    "bowl_color": bowl_color,
                    "lang_goal": lang_goal,
                    "target_block_id": int(task.target_block_id),
                    "target_bowl_id": int(task.target_bowl_id),
                    "block_ids": [int(i) for i in task.block_ids],
                    "bowl_ids": [int(i) for i in task.bowl_ids],
                    "block_id_to_color": {int(k): v for k, v in task.block_id_to_color.items()},
                    "bowl_id_to_color": {int(k): v for k, v in task.bowl_id_to_color.items()},
                    "block_attempts": int(block_attempts),
                    "bowl_attempts": int(bowl_attempts),
                    "block_click_ms": float(block_click_ms),
                    "bowl_click_ms": float(bowl_click_ms),
                    "success": success,
                    "total_reward": float(total_reward),
                }, f, indent=2)

            events.writerow([
                ep, "episode_done", "", f"{logger.relative_ms():.4f}",
                block_color, bowl_color, "", 0, str(success).lower(),
                seed, args.pair_pool,
            ])
            events_file.flush()
            print(f"[clport] episode {ep} done | success={success} reward={total_reward:.3f}")

    except KeyboardInterrupt:
        print("[clport] Interrupted; stopping.")
    finally:
        events_file.close()
        try:
            logger.stop()
        except Exception:
            pass
        try:
            env.end_rec()
        except Exception:
            pass
        try:
            p.disconnect()
        except Exception:
            pass
        print(f"[clport] Session saved to {used_session_dir}")
        print(f"[clport] {successes}/{args.session_length} successful episodes.")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="HITL CLIPort recorder.")
    parser.add_argument("--device_id", default=os.getenv("NEUROSITY_DEVICE_ID"))
    parser.add_argument("--email", default=os.getenv("NEUROSITY_EMAIL"))
    parser.add_argument("--password", default=os.getenv("NEUROSITY_PASSWORD"))
    parser.add_argument("--session_length", type=int, required=True,
                        help="Number of episodes to record.")
    parser.add_argument("--pair_pool", choices=["train", "val_unseen", "all"],
                        default="train",
                        help="Which (block,bowl) tuples are sampled this session.")
    parser.add_argument("--data_root", default="data/clport")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for reproducible episode generation.")
    parser.add_argument("--assets_root", default=None,
                        help="Path to cliport assets/. Defaults to the bundled "
                             "assets in third_party/cliport.")
    parser.add_argument("--no_eeg", action="store_true",
                        help="Skip the Neurosity Logger (sim-only smoke test).")
    args = parser.parse_args()

    if not args.no_eeg and (not args.device_id or not args.email or not args.password):
        parser.error("device_id/email/password required unless --no_eeg is set.")

    run_session(args)


if __name__ == "__main__":
    main()
