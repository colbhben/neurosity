"""Composition splits for the 4-color HITL put-block-in-bowl experiment.

The 4-color pool {blue, red, green, yellow} yields 12 ordered (block, bowl)
tuples (excluding same-color pairs). We hold out a "diagonal" of 4 tuples so
every color appears in both `train` and `val_unseen` populations, but no
specific (block, bowl) pairing is shared between them.

Public API:
    HITL_COLORS                  list[str]
    ALL_PAIRS                    list[tuple[str, str]] (12 tuples)
    HOLDOUT_UNSEEN               list[tuple[str, str]] (4 tuples)
    TRAIN_PAIRS                  list[tuple[str, str]] (8 tuples)
    color_to_idx                 dict[str, int]
    make_splits(episodes, seed)  partition Episodes into
                                 {train, val_seen, val_unseen, val_mixed}
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict
import random


HITL_COLORS: List[str] = ["blue", "red", "green", "yellow"]
ALL_PAIRS: List[Tuple[str, str]] = [
    (b, w) for b in HITL_COLORS for w in HITL_COLORS if b != w
]
HOLDOUT_UNSEEN: List[Tuple[str, str]] = [
    ("red", "blue"),
    ("blue", "green"),
    ("green", "yellow"),
    ("yellow", "red"),
]
TRAIN_PAIRS: List[Tuple[str, str]] = [pr for pr in ALL_PAIRS if pr not in HOLDOUT_UNSEEN]

color_to_idx: Dict[str, int] = {c: i for i, c in enumerate(HITL_COLORS)}


@dataclass
class Episode:
    """Lightweight handle for a recorded episode used by the splitter."""
    session_id: str
    episode_idx: int
    block_color: str
    bowl_color: str

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.block_color, self.bowl_color)


def make_splits(
    episodes: Sequence[Episode],
    seed: int = 0,
    val_seen_frac: float = 0.2,
) -> Dict[str, List[Episode]]:
    """Partition `episodes` into train / val_seen / val_unseen / val_mixed.

    - val_unseen: every episode whose pair is in HOLDOUT_UNSEEN.
    - train + val_seen: episodes whose pair is in TRAIN_PAIRS, split
      `val_seen_frac` per pair so all training pairs are represented in
      val_seen.
    - val_mixed: a 50/50 sample of val_seen + val_unseen, capped at the
      smaller pool size so the mix is balanced.
    """
    rng = random.Random(seed)

    by_pair: Dict[Tuple[str, str], List[Episode]] = {}
    for ep in episodes:
        by_pair.setdefault(ep.pair, []).append(ep)

    train: List[Episode] = []
    val_seen: List[Episode] = []
    val_unseen: List[Episode] = []
    for pair, eps in by_pair.items():
        # Stable shuffle per pair so different runs with the same seed
        # produce identical splits regardless of the input order.
        eps_sorted = sorted(eps, key=lambda e: (e.session_id, e.episode_idx))
        rng.shuffle(eps_sorted)
        if pair in HOLDOUT_UNSEEN:
            val_unseen.extend(eps_sorted)
            continue
        if pair not in TRAIN_PAIRS:
            # Pair outside the 4-color pool entirely. Drop with a warning.
            continue
        cut = max(1, int(round(len(eps_sorted) * val_seen_frac)))
        val_seen.extend(eps_sorted[:cut])
        train.extend(eps_sorted[cut:])

    n_mix = min(len(val_seen), len(val_unseen))
    mixed = []
    if n_mix > 0:
        mixed = (
            rng.sample(val_seen, n_mix)
            + rng.sample(val_unseen, n_mix)
        )
        rng.shuffle(mixed)

    return {
        "train": train,
        "val_seen": val_seen,
        "val_unseen": val_unseen,
        "val_mixed": mixed,
    }
