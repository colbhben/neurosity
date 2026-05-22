"""Item-mux cue recorder.

Drives a session of randomized item pick-up cues from N named items (default
PEN / HIGHLIGHTER) separated by reset intervals, while `log.py` records EEG to
the same session directory. Each reset assigns a unique random location index
(0..N-1) to every item and prints the assignment; the following cue picks one
of those items at random and the user mentally picks it up from its known
location. Cue events and reset events are written to `events.csv` alongside
`data.csv`, sharing the same t=0 (the first raw sample from the SDK).

The recorded events.csv has columns: episode, event, label, relative_time_ms,
location, locations. On cue rows, `label` is the item and `location` is the
slot the item currently occupies (from the most recent reset). On reset rows,
`locations` is a `;`-separated list of items in slot order (slot 0 first).

Example calls (run from the repo root, with NEUROSITY_* set in .env):

    # 20 episodes of 6s with 2s reset, default items PEN/HIGHLIGHTER.
    python workspace/record_item_mux.py \\
        --episode_length 6 --reset_length 2 --session_length 20

    # Custom item set.
    python workspace/record_item_mux.py \\
        --episode_length 6 --reset_length 2 --session_length 20 \\
        --items PEN HIGHLIGHTER ERASER

    # Quick smoke test with a fixed seed.
    python workspace/record_item_mux.py \\
        --episode_length 4 --reset_length 1.5 --session_length 5 --seed 42
"""

import argparse
import csv
import os
import random
import time
from typing import List

from dotenv import load_dotenv

from log import Logger

DEFAULT_ITEMS = ["PEN", "HIGHLIGHTER"]


def _format_layout(layout: List[str]) -> str:
    parts = [f"{i}: {name}" for i, name in enumerate(layout)]
    return "reset [" + " | ".join(parts) + "]"


def run_session(
    device_id: str,
    email: str,
    password: str,
    episode_length: float,
    reset_length: float,
    session_length: int,
    items: List[str],
    data_root: str = "data/item_mux",
    seed: int = None,
):
    if reset_length >= episode_length:
        raise ValueError("reset_length must be shorter than episode_length")
    if len(items) < 2:
        raise ValueError("need at least 2 items")
    if len(set(items)) != len(items):
        raise ValueError("item names must be unique")

    if seed is not None:
        random.seed(seed)

    logger = Logger(
        device_id=device_id,
        email=email,
        password=password,
        data_root=data_root,
    )
    print("Connecting and waiting for first sample...")
    session_dir = logger.start()
    print(f"Logging to {session_dir}")

    events_path = os.path.join(session_dir, "events.csv")
    events_file = open(events_path, "w", newline="")
    events_writer = csv.writer(events_file)
    events_writer.writerow([
        "episode", "event", "label", "relative_time_ms", "location", "locations"
    ])

    # Initial layout established before the first cue so episode 0 has known
    # item positions. Not written to events.csv (would distort timing
    # inference); each cue row carries its slot in the `location` column.
    layout = list(items)
    random.shuffle(layout)
    print(f"[1/{session_length}] " + _format_layout(layout))

    t_start = time.monotonic()

    # Each episode is structured reset-then-cue. The initial reset before
    # episode 0 is the printout above; the loop's reset block announces the
    # *next* episode (suppressed after the final cue).
    try:
        for episode in range(session_length):
            cue_target = t_start + reset_length + episode * episode_length
            _sleep_until(cue_target)
            label = random.choice(items)
            location = layout.index(label)
            rel_ms = logger.relative_ms()
            events_writer.writerow([
                episode, "cue", label, f"{rel_ms:.4f}", location, ""
            ])
            events_file.flush()
            print(f"[{episode + 1}/{session_length}] {label}")

            reset_target = cue_target + (episode_length - reset_length)
            _sleep_until(reset_target)
            random.shuffle(layout)
            rel_ms = logger.relative_ms()
            events_writer.writerow([
                episode, "reset", "", f"{rel_ms:.4f}", "", ";".join(layout)
            ])
            events_file.flush()
            if episode + 1 < session_length:
                print(f"[{episode + 2}/{session_length}] " + _format_layout(layout))

        _sleep_until(t_start + reset_length + session_length * episode_length)
    except KeyboardInterrupt:
        print("Interrupted; stopping.")
    finally:
        events_file.close()
        logger.stop()
        print(f"Session saved to {session_dir}")


def _sleep_until(target_monotonic: float):
    while True:
        remaining = target_monotonic - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Item-mux cue recorder.")
    parser.add_argument(
        "--device_id",
        default=os.getenv("NEUROSITY_DEVICE_ID"),
        help="Crown device ID (defaults to NEUROSITY_DEVICE_ID env var).",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("NEUROSITY_EMAIL"),
        help="Neurosity account email (defaults to NEUROSITY_EMAIL env var).",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEUROSITY_PASSWORD"),
        help="Neurosity account password (defaults to NEUROSITY_PASSWORD env var).",
    )
    parser.add_argument(
        "--episode_length",
        type=float,
        required=True,
        help="Seconds between item cues.",
    )
    parser.add_argument(
        "--reset_length",
        type=float,
        required=True,
        help="Seconds at the end of each episode reserved for returning to neutral.",
    )
    parser.add_argument(
        "--session_length",
        type=int,
        required=True,
        help="Number of episodes in the session.",
    )
    parser.add_argument(
        "--items",
        nargs="+",
        default=DEFAULT_ITEMS,
        help="Item names (>=2, unique). Default: PEN HIGHLIGHTER.",
    )
    parser.add_argument(
        "--data_root",
        default="data/item_mux",
        help="Root directory for session output (default: data/item_mux).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible item/location sequences.",
    )
    args = parser.parse_args()

    if not args.device_id or not args.email or not args.password:
        parser.error(
            "device_id, email, and password are required (set via flags or env)."
        )

    run_session(
        device_id=args.device_id,
        email=args.email,
        password=args.password,
        episode_length=args.episode_length,
        reset_length=args.reset_length,
        session_length=args.session_length,
        items=args.items,
        data_root=args.data_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
