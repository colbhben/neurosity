"""Left/Right cue recorder.

Drives a session of randomized LEFT/RIGHT cues separated by reset intervals,
while `log.py` records EEG to the same session directory. Cue events and reset
events are written to `events.csv` alongside `data.csv`, sharing the same t=0
(the first raw sample from the SDK).

Example calls (run from the repo root, with NEUROSITY_* set in .env):

    # 20 episodes, 6s each, last 2s reserved for returning to neutral.
    python workspace/record_lr.py \
        --episode_length 6 --reset_length 2 --session_length 20

    # Quick smoke test: 5 episodes of 4s with a fixed RNG seed.
    python workspace/record_lr.py \
        --episode_length 4 --reset_length 1.5 --session_length 5 --seed 42

    # Override credentials / device explicitly instead of using .env.
    python workspace/record_lr.py \
        --device_id 0123456789abcdef0123456789abcdef \
        --email you@example.com --password 'secret' \
        --episode_length 6 --reset_length 2 --session_length 20

    # Write sessions somewhere other than ./data.
    python workspace/record_lr.py \
        --episode_length 6 --reset_length 2 --session_length 20 \
        --data_root /tmp/neurosity_sessions
"""

import argparse
import csv
import os
import random
import time

from dotenv import load_dotenv

from log import Logger


def run_session(
    device_id: str,
    email: str,
    password: str,
    episode_length: float,
    reset_length: float,
    session_length: int,
    data_root: str = "data",
    seed: int = None,
    static_side: str = None,
):
    if reset_length >= episode_length:
        raise ValueError("reset_length must be shorter than episode_length")

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
    events_writer.writerow(["episode", "event", "label", "relative_time_ms"])

    # Anchor pacing on monotonic time but record event timestamps via the
    # logger's relative_ms() so they share t=0 with the EEG CSV.
    t_start = time.monotonic()

    try:
        for episode in range(session_length):
            cue_target = t_start + episode * episode_length
            _sleep_until(cue_target)
            label = static_side if static_side else random.choice(["LEFT", "RIGHT"])
            rel_ms = logger.relative_ms()
            events_writer.writerow([episode, "cue", label, f"{rel_ms:.4f}"])
            events_file.flush()
            shown = "MOVE" if static_side else label
            print(f"[{episode + 1}/{session_length}] {shown}")

            reset_target = cue_target + (episode_length - reset_length)
            _sleep_until(reset_target)
            rel_ms = logger.relative_ms()
            events_writer.writerow([episode, "reset", "", f"{rel_ms:.4f}"])
            events_file.flush()
            print(f"[{episode + 1}/{session_length}] reset")

        _sleep_until(t_start + session_length * episode_length)
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
    parser = argparse.ArgumentParser(description="LEFT/RIGHT cue recorder.")
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
        help="Seconds between LEFT/RIGHT cues.",
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
        "--data_root",
        default="data",
        help="Root directory for session output (default: data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible LEFT/RIGHT sequences.",
    )
    parser.add_argument(
        "--static_side",
        choices=["LEFT", "RIGHT"],
        default=None,
        help="If set, every episode is logged as this side and 'MOVE' is "
             "printed instead of LEFT/RIGHT.",
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
        data_root=args.data_root,
        seed=args.seed,
        static_side=args.static_side,
    )


if __name__ == "__main__":
    main()
