"""Launch local single-device GOT-10k training for the MVT tracking model.

Run from the repository root after `pip install -e '.[test]'`:
    python train.py --smoke-test --batch-size 2 --num-workers 0
    python train.py --batch-size 8 --grad-accum 16 --amp
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mvt.data.sampler import GOT10KPairDataset
from mvt.losses.mvt_loss import MVTLoss
from mvt.models.mvt import MVT
from mvt.training.optimizer import OptimizerConfig, build_optimizer_scheduler
from mvt.training.pretrained import load_training_weights
from mvt.training.trainer import run_epoch, save_training_checkpoint, resume_training_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MVT visual object tracker on GOT-10k")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--train-root", default="data/train")
    parser.add_argument("--val-root", default="data/val")
    parser.add_argument("--test-root", default="data/test", help="Reserved for later sequence-level benchmark evaluation; not used for training")
    parser.add_argument("--output-dir", default="checkpoints/mvt_got10k")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision")
    parser.add_argument("--resume", help="Resume from this package's latest.pth or epoch_XXXX.pth")
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument("--pretrained-backbone", help="Audited upstream MobileViT-S classification or backbone checkpoint")
    weight_group.add_argument("--pretrained-tracker", help="Native or audited-upstream MVT tracker checkpoint")
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Train and validate a few batches for 1 epoch")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resume and (args.pretrained_backbone or args.pretrained_tracker):
        raise ValueError("Use either --resume or --pretrained-*, not both")
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    c = cfg["training"]
    data = cfg["data"]
    batch_size = args.batch_size or int(c["batch_size"])
    epochs = args.epochs or int(c["epochs"])
    train_samples = args.train_samples or int(data["train"]["samples_per_epoch"])
    val_samples = args.val_samples or int(data["validation"]["samples_per_epoch"])
    workers = int(c["workers"] if args.num_workers is None else args.num_workers)
    val_interval = args.val_interval
    if args.smoke_test:
        batch_size = min(batch_size, 2)
        epochs = 1
        workers = 0
        train_samples = min(train_samples, 4)
        val_samples = min(val_samples, 2)
        val_interval = 1
    if min(batch_size, epochs, train_samples, val_samples, args.grad_accum, val_interval, args.save_every) <= 0:
        raise ValueError("Batch size, epochs, sample counts, accumulation and save/validation intervals must all be positive")
    if workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False. Install matching NVIDIA drivers and CUDA-enabled PyTorch.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"MVT | device={device} | batch={batch_size} | accumulation={args.grad_accum} | effective batch={batch_size * args.grad_accum} | epochs={epochs}", flush=True)
    print(f"Train={args.train_root} | Val={args.val_root} | Test={args.test_root} (held out; no evaluation runner yet)", flush=True)
    train_ds = GOT10KPairDataset(
        args.train_root, samples_per_epoch=train_samples,
        max_sample_interval=int(data["maximum_sample_interval"]), training=True, seed=args.seed,
    )
    val_ds = GOT10KPairDataset(
        args.val_root, samples_per_epoch=val_samples,
        max_sample_interval=int(data["maximum_sample_interval"]), training=False, seed=args.seed + 100000,
    )
    print(f"Indexed train sequences={len(train_ds.sequences)}, validation sequences={len(val_ds.sequences)}", flush=True)

    pin_memory = device.type == "cuda"
    loader_opts = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_opts)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_opts)

    model = MVT().to(device)
    if args.pretrained_backbone:
        print(load_training_weights(model, args.pretrained_backbone, backbone_only=True), flush=True)
    if args.pretrained_tracker:
        print(load_training_weights(model, args.pretrained_tracker, backbone_only=False), flush=True)
    criterion = MVTLoss(
        l1_weight=float(c["losses"]["l1_weight"]),
        giou_weight=float(c["losses"]["giou_weight"]),
        upstream_heatmap_bug_compatible=bool(c["losses"].get("upstream_heatmap_bug_compatible", False)),
    ).to(device)
    opt_config = OptimizerConfig(
        learning_rate=float(c["learning_rate"]),
        weight_decay=float(c["weight_decay"]),
        backbone_multiplier=float(c["backbone_multiplier"]),
        epochs=epochs,
        scheduler=c["scheduler"]["type"],
        eta_min=float(c["scheduler"].get("eta_min", 1e-9)),
    )
    optimizer, scheduler = build_optimizer_scheduler(model, opt_config)
    amp = bool(args.amp or c.get("amp", False)) and device.type == "cuda"
    # torch.cuda.amp.GradScaler retains compatibility with older supported torch 2.x.
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    start_epoch, best_val = 1, float("inf")
    if args.resume:
        start_epoch, best_val = resume_training_checkpoint(
            args.resume, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        )
        print(f"Resuming at epoch {start_epoch}; best validation loss={best_val:.4f}", flush=True)
    if start_epoch > epochs:
        print("Nothing to train: checkpoint already completed the requested number of epochs.")
        return

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===", flush=True)
        lr = [group["lr"] for group in optimizer.param_groups]
        train_stats = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            gradient_clip_norm=float(c["gradient_clip_norm"]),
            accumulation_steps=args.grad_accum, amp=amp, scaler=scaler,
            log_interval=args.log_interval,
        )
        val_stats = None
        if epoch % val_interval == 0 or epoch == epochs:
            val_stats = run_epoch(
                model, val_loader, criterion, device, amp=amp,
                log_interval=args.log_interval,
            )
        scheduler.step()
        entry = {"epoch": epoch, "train": train_stats, "val": val_stats, "lr": lr}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry) + "\n")
        if val_stats is not None and val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_training_checkpoint(output / "best.pth", model=model, optimizer=optimizer, scheduler=scheduler,
                                     epoch=epoch, best_val_loss=best_val, scaler=scaler)
        save_training_checkpoint(output / "latest.pth", model=model, optimizer=optimizer, scheduler=scheduler,
                                 epoch=epoch, best_val_loss=best_val, scaler=scaler)
        if epoch % args.save_every == 0 or epoch == epochs:
            save_training_checkpoint(output / f"epoch_{epoch:04d}.pth", model=model, optimizer=optimizer, scheduler=scheduler,
                                     epoch=epoch, best_val_loss=best_val, scaler=scaler)
        print(f"Epoch {epoch}: train loss={train_stats['loss']:.4f}; val loss={val_stats['loss'] if val_stats else 'not run'}; best={best_val:.4f}", flush=True)
    print(f"Training finished. Checkpoints: {output.resolve()}")


if __name__ == "__main__":
    main()
