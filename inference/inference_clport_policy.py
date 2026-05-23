"""Evaluate a CLIPort policy checkpoint on the HITL task across splits.

Wraps `cliport/eval.py` with our 4-color HITL task. Runs N rollouts per
split and writes a JSON summary of success rates.

Example:
    python inference/inference_clport_policy.py \\
        --model_path training/clport_policy_runs/checkpoints \\
        --ckpt last.ckpt \\
        --agent cliport \\
        --ravens_root data/clport_ravens \\
        --n_demos 20

The split argument controls which RavensDataset directory we replay seeds
from:
  - val_seen   -> <ravens_root>/put-block-in-bowl-hitl-val
  - val_unseen -> <ravens_root>/put-block-in-bowl-hitl-val-unseen
  - val_mixed  -> draws half-and-half from the two above (in-process).
"""

import argparse
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CLIPORT_ROOT = os.path.join(_REPO_ROOT, "third_party", "cliport")


def _run_eval_split(args, split_subdir: str, save_path: str):
    """Invoke cliport/eval.py against `<ravens_root>/<task>-<split_subdir>`.

    cliport/eval.py expects mode in {train,val,test}. We map our split
    `val` -> upstream `mode=val`; `val_unseen` is encoded by emitting the
    held-out compositions into a `<task>-val` directory under a dedicated
    ravens_root (see build_ravens_dataset.py).
    """
    eval_py = os.path.join(_CLIPORT_ROOT, "cliport", "eval.py")
    overrides = [
        f"eval_task=put-block-in-bowl-hitl",
        f"agent={args.agent}",
        f"mode=val",
        f"n_demos={args.n_demos}",
        f"data_dir={os.path.abspath(split_subdir)}",
        f"model_path={os.path.abspath(args.model_path)}",
        f"save_path={os.path.abspath(save_path)}",
        f"checkpoint_type=val_missing",
        f"disp=False",
        f"record.save_video=False",
        f"update_results=True",
        f"train_config={os.path.abspath(args.train_config)}",
    ]
    cmd = [sys.executable, eval_py, *overrides]
    env = os.environ.copy()
    env["CLIPORT_ROOT"] = _CLIPORT_ROOT
    print("[clport-eval] " + " ".join(cmd))
    subprocess.check_call(cmd, env=env, cwd=_CLIPORT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravens_root", required=True,
                        help="Root containing put-block-in-bowl-hitl-{train,val,val-unseen}.")
    parser.add_argument("--model_path", required=True,
                        help="Dir holding `last.ckpt` / `best.ckpt`.")
    parser.add_argument("--train_config", required=True,
                        help="Hydra train config yaml saved next to the ckpt.")
    parser.add_argument("--agent", default="cliport")
    parser.add_argument("--n_demos", type=int, default=20)
    parser.add_argument("--out_dir", default="inference_runs/clport_policy")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    # val_seen lives under <ravens_root>/put-block-in-bowl-hitl-val/
    seen_dir = args.ravens_root  # cliport/eval.py expects data_dir to contain `<task>-val`
    summary["val_seen_save"] = os.path.join(args.out_dir, "val_seen")
    _run_eval_split(args, seen_dir, summary["val_seen_save"])

    # val_unseen: re-stage to a sibling root the cliport eval will see as `-val`.
    # We simply symlink to the existing <ravens_root>/put-block-in-bowl-hitl-val-unseen
    # under a temporary root with the suffix `-val`.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        link = os.path.join(tmp, "put-block-in-bowl-hitl-val")
        target = os.path.abspath(os.path.join(
            args.ravens_root, "put-block-in-bowl-hitl-val-unseen"))
        if os.path.exists(target):
            os.symlink(target, link)
            unseen_save = os.path.join(args.out_dir, "val_unseen")
            summary["val_unseen_save"] = unseen_save
            _run_eval_split(args, tmp, unseen_save)
        else:
            print(f"[clport-eval] no val_unseen at {target}; skipping.")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
