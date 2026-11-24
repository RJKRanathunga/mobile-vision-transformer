
"""Extract MVT network weights without importing legacy lib modules."""

from collections.abc import Mapping
from pathlib import Path

import torch


SOURCE = Path(
    "pretrained_models/MobileViT_Track_ep0100.pth.tar"
)
OUTPUT = Path(
    "pretrained_models/MobileViT_Track_ep0100_weights_only.pth"
)


class LegacyMetadata:
    """Placeholder for unused original MVT training metadata."""

    def __init__(self, *args, **kwargs):
        pass


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    # Inspect serialized references without importing them.
    globals_found = (
        torch.serialization.get_unsafe_globals_in_checkpoint(
            SOURCE
        )
    )

    print("Legacy checkpoint globals:")
    for name in globals_found:
        print(" -", name)

    # Only replace references from the original MVT package.
    unexpected = [
        name
        for name in globals_found
        if not name.startswith("lib.")
    ]

    if unexpected:
        raise RuntimeError(
            "Additional checkpoint globals need inspection: "
            + ", ".join(unexpected)
        )

    placeholders = [
        (LegacyMetadata, name)
        for name in globals_found
    ]

    # The restricted unpickler does not dynamically import
    # the original lib modules.
    with torch.serialization.safe_globals(placeholders):
        checkpoint = torch.load(
            SOURCE,
            map_location="cpu",
            weights_only=True,
        )

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a checkpoint dictionary")

    state = checkpoint.get("net")

    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint has no 'net' state dictionary")

    if not state or not all(
        isinstance(v, torch.Tensor)
        for v in state.values()
    ):
        raise TypeError("Expected tensor-only network weights")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"net": state}, OUTPUT)

    print(f"Extracted {len(state)} network tensors")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
