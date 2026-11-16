"""Single-device training, validation, and resumable checkpoint utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 0.1,
    accumulation_steps: int = 1,
    amp: bool = False,
    log_interval: int = 50,
    scaler: Any = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    training = optimizer is not None
    model.train(training)
    running = {key: 0.0 for key in ("loss", "focal", "l1", "giou", "iou")}
    batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    # The native MVT criterion is float32; FP16 inference is optional on CUDA.
    amp_enabled = amp and device.type == "cuda"
    total_batches = len(loader) if max_steps is None else min(len(loader), max_steps)
    for step, batch in enumerate(loader, 1):
        if max_steps is not None and step > max_steps:
            break
        search = batch["search_image"].to(device, non_blocking=True)
        template = batch["template_image"].to(device, non_blocking=True)
        target = batch["search_bbox"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(search, template)
                # Loss in float32 helps GIoU / focal stability under autocast.
                predictions = {k: v.float() for k, v in predictions.items()}
                losses = criterion(predictions, target)
            loss = losses["loss"]
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            if training:
                # Correctly scale the final incomplete accumulation window.
                window_size = min(accumulation_steps, total_batches - ((step - 1) // accumulation_steps) * accumulation_steps)
                scaled_loss = loss / window_size
                if scaler is not None and amp_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if step % accumulation_steps == 0 or step == total_batches:
                    if scaler is not None and amp_enabled:
                        scaler.unscale_(optimizer)
                    if gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    if scaler is not None and amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        for key in running:
            running[key] += float(losses[key].detach().item())
        batches += 1
        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == total_batches):
            mode = "train" if training else "val"
            print(f"[{mode}] batch {step}/{total_batches}: loss={running['loss'] / batches:.4f}, IoU={running['iou'] / batches:.4f}", flush=True)

    if batches == 0:
        raise ValueError("No batches produced by the data loader; reduce batch size or increase samples_per_epoch")
    return {name: value / batches for name, value in running.items()}


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    scaler: Any = None,
) -> None:
    """Write an atomic, fully resumable state; not only the network weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(contents, temp_path)
    os.replace(temp_path, path)


def resume_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any = None,
) -> tuple[int, float]:
    """Return (next 1-based epoch, best validation loss). Trust the file source."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    for key in ("epoch", "model", "optimizer", "scheduler"):
        if key not in checkpoint:
            raise KeyError(f"Resume checkpoint lacks required key '{key}': {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_val_loss", float("inf")))
