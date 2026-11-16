"""Read extracted GOT-10k video sequences from data/train or data/val.

Each sequence contains numbered JPEG frames, groundtruth.txt in xywh format,
and (optionally) absence.label and cover.label. The official test split is
usually unlabelled and is intentionally not accepted as a training split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class GOT10KSequence:
    name: str
    frames: tuple[Path, ...]
    boxes: np.ndarray  # [N,4], unnormalized xywh
    visible_ids: tuple[int, ...]
    search_ids: tuple[int, ...]  # indices into visible_ids, with an earlier template


def _read_numeric_rows(path: Path, *, delimiter: str | None = None) -> np.ndarray:
    try:
        values = np.loadtxt(path, delimiter=delimiter, dtype=np.float32, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Could not parse annotation file {path}: {exc}") from exc
    return values


def _sequence_names(root: Path) -> list[str]:
    listing = root / "list.txt"
    if listing.is_file():
        names = [line.strip().strip("/") for line in listing.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not names:
        raise ValueError(f"No GOT-10k sequence directories found in {root}. Extract the dataset archives first.")
    return names


def discover_got10k(root: str | Path, *, max_gap: int = 200, require_annotations: bool = True) -> list[GOT10KSequence]:
    """Index a GOT-10k split; require >=2 visible annotated frames for training."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {root}. Expected data/train or data/val.")
    if max_gap <= 0:
        raise ValueError("max_gap must be positive")

    sequences: list[GOT10KSequence] = []
    errors: list[str] = []
    for name in _sequence_names(root):
        folder = root / name
        if not folder.is_dir():
            errors.append(f"listed sequence directory missing: {folder}")
            continue
        frames = tuple(sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem)))
        if not frames:
            errors.append(f"no image frames: {folder}")
            continue

        gt_path = folder / "groundtruth.txt"
        if not gt_path.is_file():
            if require_annotations:
                errors.append(f"missing groundtruth.txt: {folder}")
            continue
        boxes = _read_numeric_rows(gt_path, delimiter=",")
        if boxes.shape[1] != 4:
            errors.append(f"expected xywh boxes (4 columns), got {boxes.shape[1]} in {gt_path}")
            continue
        if boxes.shape[0] != len(frames):
            errors.append(f"frame/box count differs for {name}: {len(frames)} images vs {len(boxes)} annotations")
            continue

        visible = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        absence_path, cover_path = folder / "absence.label", folder / "cover.label"
        for path, is_absence in ((absence_path, True), (cover_path, False)):
            if path.is_file():
                values = _read_numeric_rows(path).reshape(-1)
                if len(values) != len(frames):
                    errors.append(f"label/frame count differs in {path}")
                    visible = np.zeros(len(frames), dtype=bool)
                    break
                visible &= (values == 0) if is_absence else (values > 0)
        visible_ids = tuple(int(i) for i in np.flatnonzero(visible))
        # Search must occur after a visible template and within the causal gap.
        eligible = tuple(j for j in range(1, len(visible_ids)) if visible_ids[j] - visible_ids[j - 1] <= max_gap)
        if not eligible:
            errors.append(f"no usable causal template/search pairs: {name}")
            continue
        sequences.append(GOT10KSequence(name, frames, boxes, visible_ids, eligible))

    if not sequences:
        detail = "\n".join(errors[:8]) or "No annotated, visible frame pairs found."
        raise ValueError(f"No usable GOT-10k sequences in {root}.\n{detail}")
    if errors:
        print(f"[GOT-10k] {len(errors)} sequence(s) skipped in {root}; first reason: {errors[0]}")
    return sequences


def inspect_split(root: str | Path) -> dict[str, int | str]:
    """Report an extracted split, even if the official test split has no labels."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {"path": str(root), "sequences": 0, "images": 0, "annotated": 0, "exists": 0}
    try:
        names = _sequence_names(root)
    except ValueError:
        return {"path": str(root), "exists": 1, "sequences": 0, "images": 0, "annotated": 0}
    sequences = [root / name for name in names if (root / name).is_dir()]
    return {
        "path": str(root), "exists": 1, "sequences": len(sequences),
        "images": sum(sum(p.suffix.lower() in _IMAGE_EXTENSIONS for p in folder.iterdir() if p.is_file()) for folder in sequences),
        "annotated": sum((folder / "groundtruth.txt").is_file() for folder in sequences),
    }
