"""10-10 / 10-20 electrode-name lookup for LaBraM's channel embedding.

LaBraM's `NeuralTransformer.forward(x, input_chans=...)` expects `input_chans`
as a list of indices into its learned channel-position embedding table. The
table is built from the `standard_1020` list in `third_party/labram/utils.py`.
Index 0 is reserved for the CLS token, so an electrode at position `i` in the
list corresponds to embedding index `i + 1`.

We re-export the same list here so the rest of the harness doesn't need to
import LaBraM directly during config parsing or unit tests. Keep this list
verbatim with `third_party/labram/utils.py::standard_1020`.
"""

from typing import List

# Verbatim copy of third_party/labram/utils.py::standard_1020 (commit c431221).
STANDARD_1020: List[str] = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

_INDEX = {name: i for i, name in enumerate(STANDARD_1020)}


def channel_to_embedding_index(name: str) -> int:
    """Return the LaBraM embedding index for a single 10-10 electrode name.

    Adds 1 to skip the CLS slot (`pos_embed[:, 0]`).
    """
    key = name.upper()
    if key not in _INDEX:
        raise KeyError(
            f"Channel {name!r} is not in LaBraM's standard_1020 list. "
            f"Add it to third_party/labram/utils.py::standard_1020 first."
        )
    return _INDEX[key] + 1


def input_chans_for(channels: List[str]) -> List[int]:
    """Build LaBraM `input_chans` argument including the CLS slot at index 0."""
    return [0] + [channel_to_embedding_index(c) for c in channels]
