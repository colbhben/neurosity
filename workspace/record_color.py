"""Color attention cue recorder.

Drives a session of randomized color cues using a daisy-chained WS2812 matrix
(see `third_party/ledmatrix`). Each episode follows:

  1. cue phase   (default 0.5s): all LEDs lit in the cued color at
     `--brightness` (default 5%/255 ≈ 13).
  2. target phase (default 1.0s): one unique random LED is lit per color in
     `--colors`, simultaneously, at `--target_brightness` (default 255 = 100%).
     The user fixates on the LED matching the cued color from step 1.
  3. reset phase (default 0.5s): all LEDs off; subject relaxes.

EEG is logged via `log.py` (same t=0 as `events.csv`). Each cue row includes
the cue color, the (x,y) of the cued LED in the target phase, and the (x,y)
of each color, so the data is reusable for spatial-attention style decoders.

Example calls (run from the repo root, with NEUROSITY_* set in .env):

    # 30 episodes, default 0.5s/1.0s/0.5s cue/target/reset.
    python workspace/record_color.py --port /dev/cu.usbmodem1234 \\
        --session_length 30

    # Custom timing and color set.
    python workspace/record_color.py --port /dev/cu.usbmodem1234 \\
        --session_length 40 --cue_length 0.5 --target_length 1.0 \\
        --reset_length 0.5 --colors RED GREEN BLUE YELLOW
"""

import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from log import Logger

# Allow `from ledmatrix import ...` without installing the submodule.
_LEDMATRIX_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "ledmatrix", "src"
)
if os.path.isdir(_LEDMATRIX_SRC) and _LEDMATRIX_SRC not in sys.path:
    sys.path.insert(0, _LEDMATRIX_SRC)

from ledmatrix import Layout, MatrixDriver, PanelOrientation  # noqa: E402
from ledmatrix.layout import ChainOrder  # noqa: E402


# Base unit-RGB; scaled at runtime by --brightness so 5% = 5/100 of 255.
COLOR_UNIT_RGB: Dict[str, Tuple[float, float, float]] = {
    "RED": (1.0, 0.0, 0.0),
    "GREEN": (0.0, 1.0, 0.0),
    "BLUE": (0.0, 0.0, 1.0),
    "YELLOW": (1.0, 1.0, 0.0),
}
DEFAULT_COLORS = ["RED", "GREEN", "BLUE", "YELLOW"]


def _scale(unit: Tuple[float, float, float], brightness: int) -> Tuple[int, int, int]:
    return (
        int(round(unit[0] * brightness)),
        int(round(unit[1] * brightness)),
        int(round(unit[2] * brightness)),
    )


def _build_layout() -> Layout:
    """32x16 stack of two 8x32 panels — same wiring as ledmatrix smoke_test."""
    return Layout(
        panels_x=1,
        panels_y=2,
        panel_w=32,
        panel_h=8,
        panel_orientation=PanelOrientation.DEG_0,
        chain_order=ChainOrder.VERTICAL_PROGRESSIVE,
        global_rotation=0,
        chain_reversed=True,
        flip_alternate_panels=True,
    )


def _sleep_until(target_monotonic: float):
    while True:
        remaining = target_monotonic - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def _format_positions(positions: Dict[str, Tuple[int, int]]) -> str:
    """Encode cue-row position metadata as 'RED:1,2;GREEN:3,4;...'."""
    return ";".join(f"{c}:{x},{y}" for c, (x, y) in positions.items())


def run_session(
    *,
    device_id: str,
    email: str,
    password: str,
    port: str,
    cue_length: float,
    target_length: float,
    reset_length: float,
    session_length: int,
    colors: List[str],
    brightness: int,
    target_brightness: int,
    layout: Layout,
    data_root: str = "data/color",
    seed: int = None,
):
    if cue_length <= 0 or target_length <= 0 or reset_length <= 0:
        raise ValueError("cue_length, target_length, reset_length must all be > 0")
    if any(c not in COLOR_UNIT_RGB for c in colors):
        raise ValueError(f"unknown color in {colors}; supported: {list(COLOR_UNIT_RGB)}")
    if len(set(colors)) != len(colors):
        raise ValueError("--colors must be unique")
    if len(colors) < 2:
        raise ValueError("need at least 2 colors")
    if brightness < 1 or brightness > 255:
        raise ValueError("--brightness must be in [1, 255]")
    if target_brightness < 1 or target_brightness > 255:
        raise ValueError("--target_brightness must be in [1, 255]")
    if layout.total_pixels < len(colors):
        raise ValueError(
            f"layout has {layout.total_pixels} pixels; need at least {len(colors)} for unique target LEDs"
        )

    if seed is not None:
        random.seed(seed)

    cue_rgb = {c: _scale(COLOR_UNIT_RGB[c], brightness) for c in colors}
    target_rgb = {c: _scale(COLOR_UNIT_RGB[c], target_brightness) for c in colors}
    all_positions = list(layout.iter_xy())

    print(f"Display: {layout.width}x{layout.height} ({layout.total_pixels} LEDs)")
    print(
        f"Trial: cue={cue_length}s target={target_length}s reset={reset_length}s "
        f"colors={colors} brightness={brightness}/255 target_brightness={target_brightness}/255"
    )

    logger = Logger(
        device_id=device_id, email=email, password=password, data_root=data_root
    )
    print("Connecting and waiting for first sample...")
    session_dir = logger.start()
    print(f"Logging to {session_dir}")

    events_path = os.path.join(session_dir, "events.csv")
    events_file = open(events_path, "w", newline="")
    events_writer = csv.writer(events_file)
    events_writer.writerow([
        "episode",
        "event",
        "label",
        "relative_time_ms",
        "cue_x",
        "cue_y",
        "positions",
    ])

    episode_length = cue_length + target_length + reset_length

    try:
        with MatrixDriver(port, layout) as drv:
            if not drv.ping():
                raise IOError("LED driver did not respond to ping")
            drv.clear()
            drv.show()

            # Anchor pacing after the Arduino's ~2s bootloader wait + ping +
            # initial clear so episode 0 doesn't get a zero-length target
            # phase from cue_target/target_target landing in the past.
            t_start = time.monotonic()

            for episode in range(session_length):
                # ---- cue phase: full board lit in cue color ------------------
                cue_target = t_start + episode * episode_length
                _sleep_until(cue_target)

                cue_color = random.choice(colors)
                drv.fill(cue_rgb[cue_color])
                drv.show()

                cue_ms = logger.relative_ms()

                # Sample unique target LEDs while the cue is shown so the
                # transition between cue and target phases is tight.
                positions = dict(
                    zip(colors, random.sample(all_positions, len(colors)))
                )
                cue_pos = positions[cue_color]
                events_writer.writerow([
                    episode,
                    "cue",
                    cue_color,
                    f"{cue_ms:.4f}",
                    cue_pos[0],
                    cue_pos[1],
                    _format_positions(positions),
                ])
                events_file.flush()
                print(
                    f"[{episode + 1}/{session_length}] cue={cue_color} "
                    f"target_xy=({cue_pos[0]},{cue_pos[1]})"
                )

                # ---- target phase: one LED per color -------------------------
                # Target onset is deterministic at cue_ms + cue_length; we
                # don't emit a separate event row so events.csv matches the
                # cue/reset cadence consumed by the rest of the harness.
                target_target = cue_target + cue_length
                _sleep_until(target_target)
                drv.clear()
                for c, (x, y) in positions.items():
                    drv.set_pixel(x, y, target_rgb[c])
                drv.show()

                # ---- reset phase: dark --------------------------------------
                reset_target = target_target + target_length
                _sleep_until(reset_target)
                drv.clear()
                drv.show()
                reset_ms = logger.relative_ms()
                events_writer.writerow([
                    episode, "reset", "", f"{reset_ms:.4f}", "", "", "",
                ])
                events_file.flush()

            _sleep_until(t_start + session_length * episode_length)
    except KeyboardInterrupt:
        print("Interrupted; stopping.")
    finally:
        events_file.close()
        logger.stop()
        print(f"Session saved to {session_dir}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Color attention cue recorder.")
    parser.add_argument("--device_id", default=os.getenv("NEUROSITY_DEVICE_ID"))
    parser.add_argument("--email", default=os.getenv("NEUROSITY_EMAIL"))
    parser.add_argument("--password", default=os.getenv("NEUROSITY_PASSWORD"))
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port for the LED matrix Arduino (e.g. /dev/cu.usbmodem1234).",
    )
    parser.add_argument(
        "--cue_length", type=float, default=0.5,
        help="Seconds the full board is lit in the cue color (default 0.5).",
    )
    parser.add_argument(
        "--target_length", type=float, default=1.0,
        help="Seconds the per-color single-LED display is shown (default 1.0).",
    )
    parser.add_argument(
        "--reset_length", type=float, default=0.5,
        help="Seconds the board is dark between episodes (default 0.5).",
    )
    parser.add_argument(
        "--session_length", type=int, required=True,
        help="Number of episodes in the session.",
    )
    parser.add_argument(
        "--colors", nargs="+", default=DEFAULT_COLORS,
        help=f"Colors to use (default: {' '.join(DEFAULT_COLORS)}). "
        f"Supported: {' '.join(COLOR_UNIT_RGB)}.",
    )
    parser.add_argument(
        "--brightness", type=int, default=13,
        help="Per-channel max brightness for the cue (full-board) phase, "
             "0-255. Default 13 ≈ 5%% of 256.",
    )
    parser.add_argument(
        "--target_brightness", type=int, default=255,
        help="Per-channel max brightness for the per-color single-LED target "
             "phase, 0-255. Default 255 (100%%).",
    )
    parser.add_argument(
        "--data_root", default="data/color",
        help="Root directory for session output (default: data/color).",
    )
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if not args.device_id or not args.email or not args.password:
        parser.error(
            "device_id, email, and password are required (set via flags or env)."
        )

    layout = _build_layout()
    run_session(
        device_id=args.device_id,
        email=args.email,
        password=args.password,
        port=args.port,
        cue_length=args.cue_length,
        target_length=args.target_length,
        reset_length=args.reset_length,
        session_length=args.session_length,
        colors=args.colors,
        brightness=args.brightness,
        target_brightness=args.target_brightness,
        layout=layout,
        data_root=args.data_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
