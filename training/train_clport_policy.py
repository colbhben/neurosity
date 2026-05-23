"""Thin wrapper around CLIPort's training loop for the HITL task.

CLIPort's Hydra-based training script (`cliport/train.py`) expects data in
RavensDataset format. This wrapper:
  1. Builds the RavensDataset directory tree from HITL session data
     (using `training.clport.build_ravens_dataset`).
  2. Invokes `cliport.train.main` with the right Hydra overrides.

For the language-only baseline this is a one-shot:
    python training/train_clport_policy.py \\
        --session_dirs data/clport/2026-05-22_* \\
        --agent cliport \\
        --n_demos 60 --n_steps 10000

For the EEG-conditioned variant, pass `--with_eeg` to also emit `eeg.pkl`
sidecars and `--agent two_stream_clip_lingunet_lat_transporter_eeg` to use
our EEG-augmented agent (registered in our cliport fork; see
[cliport/agents/eeg_conditioned.py]).

Note: this wrapper only orchestrates. Hyperparameters that need tuning
(learning rate, n_rotations, etc.) live in
`third_party/cliport/cliport/cfg/train.yaml`.
"""

import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CLIPORT_ROOT = os.path.join(_REPO_ROOT, "third_party", "cliport")


def _build_ravens(args):
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "training", "clport", "build_ravens_dataset.py"),
        "--session_dirs", *args.session_dirs,
        "--out_root", args.ravens_root,
        "--task_name", "put-block-in-bowl-hitl",
    ]
    if args.success_only:
        cmd.append("--success_only")
    if args.with_eeg:
        cmd.append("--with_eeg")
        cmd.extend(["--eeg_window_seconds", str(args.eeg_window_seconds)])
    print("[clport] " + " ".join(cmd))
    subprocess.check_call(cmd)


def _run_cliport_train(args):
    train_py = os.path.join(_CLIPORT_ROOT, "cliport", "train.py")
    overrides = [
        f"train.task=put-block-in-bowl-hitl",
        f"train.agent={args.agent}",
        f"train.n_demos={args.n_demos}",
        f"train.n_steps={args.n_steps}",
        f"train.n_val={args.n_val}",
        f"train.data_dir={os.path.abspath(args.ravens_root)}",
        f"train.train_dir={os.path.abspath(args.train_dir)}",
        f"train.log=False",
        f"dataset.cache=False",
    ]
    cmd = [sys.executable, train_py, *overrides]
    env = os.environ.copy()
    env["CLIPORT_ROOT"] = _CLIPORT_ROOT
    print("[clport] " + " ".join(cmd))
    subprocess.check_call(cmd, env=env, cwd=_CLIPORT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dirs", nargs="+", required=True,
                        help="HITL session dirs (data/clport/<id>).")
    parser.add_argument("--ravens_root", default="data/clport_ravens",
                        help="Where to materialize RavensDataset format.")
    parser.add_argument("--train_dir", default="training/clport_policy_runs",
                        help="CLIPort train output dir (checkpoints).")
    parser.add_argument("--agent", default="cliport",
                        help="CLIPort agent name. See "
                             "third_party/cliport/cliport/agents/__init__.py.")
    parser.add_argument("--n_demos", type=int, required=True)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--n_val", type=int, default=10)
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--with_eeg", action="store_true")
    parser.add_argument("--eeg_window_seconds", type=float, default=8.0)
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip the HITL -> Ravens conversion (reuse existing).")
    args = parser.parse_args()

    if not args.skip_build:
        _build_ravens(args)
    _run_cliport_train(args)


if __name__ == "__main__":
    main()
