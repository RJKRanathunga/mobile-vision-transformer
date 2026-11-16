"""Causal template/search sampling and preprocessing for GOT-10k."""
from __future__ import annotations

import bisect
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mvt.data.augmentations import augment_pair
from mvt.data.got10k import GOT10KSequence, discover_got10k
from mvt.data.input import MVTInputProcessor
from mvt.data.processing import MVTProcessing


class GOT10KPairDataset(Dataset):
    """Samples two visible frames from a sequence, template before search.

    Returns image tensors and normalized xywh boxes matching MVT/MVTLoss.
    The source root is exactly one split, e.g. data/train or data/val.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        samples_per_epoch: int,
        max_sample_interval: int = 200,
        training: bool = True,
        seed: int = 2026,
        max_retries: int = 32,
    ) -> None:
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        self.samples_per_epoch = samples_per_epoch
        self.training = training
        self.seed = seed
        self.max_gap = max_sample_interval
        self.max_retries = max_retries
        self.sequences: list[GOT10KSequence] = discover_got10k(root, max_gap=max_sample_interval)
        self.processing = MVTProcessing()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample(self, rng: random.Random) -> tuple[GOT10KSequence, int, int]:
        seq = rng.choice(self.sequences)
        search_pos = rng.choice(seq.search_ids)
        search_frame = seq.visible_ids[search_pos]
        first_pos = bisect.bisect_left(seq.visible_ids, search_frame - self.max_gap)
        template_pos = rng.randrange(first_pos, search_pos)
        return seq, seq.visible_ids[template_pos], search_frame

    def _get_item(self, rng: random.Random) -> dict[str, torch.Tensor]:
        for _ in range(self.max_retries):
            seq, template_id, search_id = self._sample(rng)
            template_box = torch.tensor(seq.boxes[template_id].copy(), dtype=torch.float32)
            search_box = torch.tensor(seq.boxes[search_id].copy(), dtype=torch.float32)
            template_image = MVTInputProcessor.load_image(seq.frames[template_id])
            search_image = MVTInputProcessor.load_image(seq.frames[search_id])
            processed = self.processing(template_image, template_box, search_image, search_box)
            box = processed["search_bbox"]
            center = box[:2] + 0.5 * box[2:]
            if not (torch.isfinite(box).all() and (box[2:] > 0).all() and (center >= 0).all() and (center < 1).all()):
                continue
            return augment_pair(
                processed["template_image"], processed["template_bbox"],
                processed["search_image"], processed["search_bbox"],
                training=self.training, rng=rng,
            )
        raise RuntimeError("Could not sample a usable tracking pair after 32 attempts. Check bbox annotations and crop jitter.")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.training:
            return self._get_item(random)
        # Stable validation sample and crop jitter for each integer index.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + index)
            return self._get_item(random.Random(self.seed + index))
