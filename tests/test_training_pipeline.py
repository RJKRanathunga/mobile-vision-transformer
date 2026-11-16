"""Small synthetic end-to-end integration checks; no GOT-10k download needed."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mvt.data.got10k import discover_got10k, inspect_split
from mvt.data.sampler import GOT10KPairDataset
from mvt.losses.mvt_loss import MVTLoss
from mvt.models.mvt import MVT
from mvt.training.trainer import run_epoch, save_training_checkpoint, resume_training_checkpoint
from mvt.training.optimizer import OptimizerConfig, build_optimizer_scheduler
from torch.utils.data import DataLoader


def _make_split(root: Path):
    folder = root / "GOT-10k_Train_000001"
    folder.mkdir(parents=True)
    root.joinpath("list.txt").write_text(folder.name + "\n", encoding="utf-8")
    boxes = np.array([[30, 30, 35, 35]] * 4, dtype=int)
    np.savetxt(folder / "groundtruth.txt", boxes, fmt="%d", delimiter=",")
    np.savetxt(folder / "absence.label", np.array([0, 0, 0, 0]), fmt="%d")
    np.savetxt(folder / "cover.label", np.array([8, 8, 8, 8]), fmt="%d")
    for i in range(1, 5):
        Image.fromarray(np.full((96, 96, 3), 80 + i, dtype=np.uint8)).save(folder / f"{i:08d}.jpg")


def test_dataset_loading_and_pair_shapes(tmp_path):
    root = tmp_path / "train"
    _make_split(root)
    assert inspect_split(root)["images"] == 4
    seqs = discover_got10k(root)
    assert len(seqs) == 1
    ds = GOT10KPairDataset(root, samples_per_epoch=2, training=False)
    sample = ds[0]
    assert sample["template_image"].shape == (3, 128, 128)
    assert sample["search_image"].shape == (3, 256, 256)
    assert sample["search_bbox"].shape == (4,)
    assert sample["search_image"].min() >= 0 and sample["search_image"].max() <= 1
    assert torch.allclose(sample["search_bbox"], ds[0]["search_bbox"])


def test_checkpoint_resume_and_backward(tmp_path):
    root = tmp_path / "train"
    _make_split(root)
    ds = GOT10KPairDataset(root, samples_per_epoch=1, training=False)
    sample = ds[0]
    model = MVT()
    model.eval()
    criterion = MVTLoss()
    optimizer, scheduler = build_optimizer_scheduler(model, OptimizerConfig(epochs=1))
    search = sample["search_image"].unsqueeze(0)
    template = sample["template_image"].unsqueeze(0)
    pred = model(search, template)
    loss = criterion(pred, sample["search_bbox"].unsqueeze(0))["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()
    cp = tmp_path / "latest.pth"
    save_training_checkpoint(cp, model=model, optimizer=optimizer, scheduler=scheduler, epoch=1, best_val_loss=3.0)
    next_epoch, best = resume_training_checkpoint(cp, model=model, optimizer=optimizer, scheduler=scheduler)
    assert next_epoch == 2 and best == 3.0
