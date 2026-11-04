from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    backbone_multiplier: float = 0.1
    epochs: int = 100
    scheduler: str = "cosine_anneal"
    decay_rate: float = 0.5
    step_size: int = 10
    eta_min: float = 1e-9


def build_optimizer_scheduler(
    model: nn.Module,
    config: OptimizerConfig | None = None,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LRScheduler]:
    """Build the optimizer groups and scheduler used by upstream MVT."""
    config = config or OptimizerConfig()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    non_backbone = [
        parameter
        for name, parameter in named_parameters
        if "backbone" not in name
    ]
    backbone = [
        parameter
        for name, parameter in named_parameters
        if "backbone" in name
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": non_backbone},
            {
                "params": backbone,
                "lr": (
                    config.learning_rate * config.backbone_multiplier
                ),
            },
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if config.scheduler == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.eta_min,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.decay_rate,
        )
    else:
        raise ValueError(f"unsupported scheduler: {config.scheduler}")
    return optimizer, scheduler
