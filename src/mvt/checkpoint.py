from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


UPSTREAM_COMMIT = "880e74b1d1f720cb51d78b68655a4ffef94ed954"


@dataclass(frozen=True)
class CheckpointLoadReport:
    format: str
    tensors_loaded: int
    upstream_commit: str | None = None


def _read_checkpoint(
    checkpoint: str | Path | Mapping[str, Any],
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        loaded = torch.load(checkpoint, map_location=map_location)
    else:
        loaded = checkpoint

    if not isinstance(loaded, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    if "net" in loaded:
        loaded = loaded["net"]
    if not isinstance(loaded, Mapping):
        raise TypeError("checkpoint['net'] must contain a state dictionary")
    return loaded


def _strip_distributed_prefix(
    state: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    return OrderedDict(
        (
            key.removeprefix("module."),
            value,
        )
        for key, value in state.items()
    )


def load_mvt_checkpoint(
    module: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointLoadReport:
    """Load either a native checkpoint or the pinned upstream MVT layout.

    The upstream and local modules have a one-to-one tensor order and shape,
    but intentionally different class and attribute names. For the pinned
    commit, order-based conversion is validated exhaustively before loading;
    any count or shape drift fails loudly instead of silently misloading data.
    """

    source = _strip_distributed_prefix(
        _read_checkpoint(checkpoint, map_location)
    )
    target = module.state_dict()

    if set(source) == set(target):
        module.load_state_dict(source, strict=True)
        return CheckpointLoadReport(
            format="native",
            tensors_loaded=len(target),
        )

    # Allow loading a full upstream tracker checkpoint into its backbone.
    backbone_only = OrderedDict(
        (key.removeprefix("backbone."), value)
        for key, value in source.items()
        if key.startswith("backbone.")
    )
    candidates = [source]
    if backbone_only:
        candidates.append(backbone_only)

    upstream_markers = (
        "backbone.conv_1.",
        "conv_1.",
        "neck.BN_x.",
    )
    if not any(key.startswith(upstream_markers) for key in source):
        raise ValueError(
            "checkpoint keys are neither native MVT keys nor the supported "
            f"upstream layout at {UPSTREAM_COMMIT}"
        )

    for candidate in candidates:
        if len(candidate) != len(target):
            continue

        converted: OrderedDict[str, torch.Tensor] = OrderedDict()
        mismatch = None
        for (source_key, source_value), (target_key, target_value) in zip(
            candidate.items(),
            target.items(),
        ):
            if not isinstance(source_value, torch.Tensor):
                mismatch = (
                    f"{source_key} is {type(source_value).__name__}, "
                    "not a tensor"
                )
                break
            if source_value.shape != target_value.shape:
                mismatch = (
                    f"{source_key} {tuple(source_value.shape)} cannot map to "
                    f"{target_key} {tuple(target_value.shape)}"
                )
                break
            converted[target_key] = source_value

        if mismatch is None:
            module.load_state_dict(converted, strict=True)
            return CheckpointLoadReport(
                format="upstream",
                tensors_loaded=len(converted),
                upstream_commit=UPSTREAM_COMMIT,
            )

    raise ValueError(
        "upstream checkpoint tensor count or shapes do not match this MVT "
        "configuration"
    )
