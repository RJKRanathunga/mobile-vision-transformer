"""Strictly validated optional pretrained weight loading for MVT."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from mvt.checkpoint import load_mvt_checkpoint


def load_training_weights(model, path: str | Path, *, backbone_only: bool = False) -> str:
    """Load a full MVT tracker or pinned-upstream MobileViT-S backbone.

    The original MobileViT-S classification checkpoint contains additional
    classifier/layer_5 weights, so we filter only the trained tracking stem
    (conv_1 and layer_1 through layer_4) and validate every tensor shape.
    Never silently load unmatched tensors.
    """
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {source_path}")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a mapping of tensors")
    state = payload
    for key in ("state_dict", "model", "net"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    state = OrderedDict((k.removeprefix("module."), v) for k, v in state.items())

    if not backbone_only:
        report = load_mvt_checkpoint(model, state)
        return f"Loaded full tracker ({report.format}, {report.tensors_loaded} tensors)"

    backbone = model.backbone
    try:
        report = load_mvt_checkpoint(backbone, state)
        return f"Loaded backbone ({report.format}, {report.tensors_loaded} tensors)"
    except ValueError:
        pass

    # Upstream MobileViT-S classification checkpoint contains the four first
    # tracker backbone layers plus extra classifier layers. Validate the
    # prefix and tensor-by-tensor order, just as the audited converter does.
    prefixes = ("conv_1.", "layer_1.", "layer_2.", "layer_3.", "layer_4.")
    filtered = [(key, value) for key, value in state.items() if key.startswith(prefixes)]
    target = list(backbone.state_dict().items())
    if len(filtered) != len(target):
        raise ValueError(f"Pretrained backbone tensor count mismatch: got {len(filtered)}, expected {len(target)}. This checkpoint may not be from the audited MobileViT-S version.")
    mapped = OrderedDict()
    for (from_name, tensor), (to_name, expected) in zip(filtered, target):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape:
            raise ValueError(f"Pretrained tensor mismatch at {from_name} -> {to_name}; expected {tuple(expected.shape)}")
        mapped[to_name] = tensor
    backbone.load_state_dict(mapped, strict=True)
    return f"Loaded pinned-upstream MobileViT-S backbone ({len(mapped)} tensors); classifier ignored"
