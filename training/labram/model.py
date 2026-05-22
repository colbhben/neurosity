"""LaBraMFinetune: NeuralTransformer backbone + per-task heads."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
from torch import nn

from training.labram.channel_map import input_chans_for
from training.labram.config import HeadConfig, TaskConfig
from training.labram.heads import build_head, head_needs_patch_tokens

# Add the LaBraM submodule to sys.path so we can import its model factory.
_LABRAM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "labram")
)
if _LABRAM_DIR not in sys.path:
    sys.path.insert(0, _LABRAM_DIR)

import modeling_finetune  # noqa: E402,F401  (registers `labram_*` factories)
from timm.models import create_model  # noqa: E402


class LaBraMFinetune(nn.Module):
    def __init__(
        self,
        channels: List[str],
        heads: List[HeadConfig],
        backbone: str = "labram_base_patch200_200",
        drop_path_rate: float = 0.1,
        use_mean_pooling: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        # `num_classes=0` makes LaBraM's stock head an Identity; we mount our own.
        self.backbone = create_model(
            backbone,
            pretrained=False,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=0.0,
            use_mean_pooling=use_mean_pooling,
            init_scale=0.001,
            use_rel_pos_bias=True,
            use_abs_pos_emb=True,
            init_values=0.1,
        )
        self.embed_dim = self.backbone.embed_dim

        self.heads = nn.ModuleDict()
        self.head_specs: Dict[str, HeadConfig] = {}
        self._wants_patch_tokens = False
        for spec in heads:
            self.heads[spec.name] = build_head(spec, self.embed_dim)
            self.head_specs[spec.name] = spec
            if head_needs_patch_tokens(spec):
                self._wants_patch_tokens = True

        self.register_buffer(
            "input_chans",
            torch.tensor(input_chans_for(channels), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run backbone once; dispatch to all heads.

        Returns `{head_name: logits_or_pred}`. If any head is a token decoder
        and `targets` is provided, we also pass the patch tokens through
        teacher-forced decoding.
        """
        # CLS pooled embedding.
        cls = self.backbone.forward_features(
            x, input_chans=self.input_chans, return_patch_tokens=False
        )  # (B, D)
        outputs: Dict[str, torch.Tensor] = {}
        patch_tokens: Optional[torch.Tensor] = None
        if self._wants_patch_tokens:
            patch_tokens = self.backbone.forward_features(
                x, input_chans=self.input_chans, return_patch_tokens=True
            )  # (B, S, D)

        for name, head in self.heads.items():
            spec = self.head_specs[name]
            if spec.type == "token":
                tgt = targets.get(name) if targets is not None else None
                outputs[name] = head(patch_tokens, tgt)
            else:
                outputs[name] = head(cls)
        return outputs

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-head losses (callers do the weighted sum)."""
        out: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            out[name] = head.loss(outputs[name], targets[name], masks[name])
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(x)
        preds: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            spec = self.head_specs[name]
            if spec.type == "token":
                # For token heads, generate sequences from patch tokens.
                patch_tokens = self.backbone.forward_features(
                    x, input_chans=self.input_chans, return_patch_tokens=True
                )
                preds[name] = head.generate(patch_tokens)
            else:
                preds[name] = head.predict(outputs[name])
        return preds


def load_pretrained(model: LaBraMFinetune, ckpt_path: str, *, strict: bool = False) -> None:
    """Load a LaBraM pretrained checkpoint into the backbone.

    LaBraM release checkpoints store weights under `model` or `module` keys and
    include a pretraining decoder we don't want. Filter to encoder weights and
    drop shape mismatches so we can load even when num_classes differs.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("model", "module", "state_dict"):
            if key in sd:
                sd = sd[key]
                break

    own_sd = model.backbone.state_dict()
    filtered = {}
    skipped = []
    for k, v in sd.items():
        # LaBraM's pretraining ckpt prefixes encoder weights with "student.".
        clean = k
        for prefix in ("student.", "encoder.", "module.", "backbone."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        if clean in own_sd and own_sd[clean].shape == v.shape:
            filtered[clean] = v
        else:
            skipped.append(k)
    missing, unexpected = model.backbone.load_state_dict(filtered, strict=False)
    print(
        f"Loaded {len(filtered)} backbone tensors from {ckpt_path}. "
        f"skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}"
    )
