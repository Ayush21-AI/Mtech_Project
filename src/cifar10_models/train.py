"""Production training loop for CIFAR-10 models."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cifar10_models.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricsLogger,
    MLFlowLogger,
    WandbLogger,
)
from cifar10_models.config import TrainConfig
from cifar10_models.distributed import get_rank, get_world_size, is_distributed
from cifar10_models.evaluate import evaluate_loader
from cifar10_models.metrics import MetricsTracker
from cifar10_models.models.model_factory import build_model
from cifar10_models.optim import AMPManager, EMAModel, get_optimizer, get_scheduler, linear_warmup_lr

logger = logging.getLogger("cifar10_models")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    amp_manager: AMPManager,
    epoch: int,
    desc: str,
    log_interval: int,
    ema: EMAModel | None = None,
    gradient_clip: float = 0.0,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    tracker = MetricsTracker()

    with torch.set_grad_enabled(is_training):
        pbar = tqdm(loader, desc=desc, leave=False)
        for step, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with amp_manager.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if is_training:
                amp_manager.scale_loss(loss).backward()

                if gradient_clip > 0:
                    amp_manager.unscale(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                amp_manager.step(optimizer)

                if ema is not None:
                    ema.update(model)

                if scheduler is not None:
                    scheduler.step()

            tracker.update(loss.item(), outputs.detach(), targets.detach(), inputs.size(0))

            if is_training and step % log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{tracker.avg_loss():.4f}",
                        "acc": f"{tracker.accuracy():.4f}",
                    }
                )

    return tracker.avg_loss(), tracker.accuracy()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    callbacks: list[Callback] | None = None,
) -> dict[str, list[float]]:
    """Train a CIFAR-10 model with production features.

    Parameters
    ----------
    model: nn.Module
        Model to train.
    train_loader: DataLoader
        Training data loader.
    val_loader: DataLoader
        Validation data loader.
    config: TrainConfig
        Training configuration.
    callbacks: list[Callback] | None
        Optional list of callbacks.

    Returns
    -------
    dict[str, list[float]]
        Training history.
    """
    device = config.device
    model = model.to(device)
    if config.compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    optimizer = get_optimizer(
        model,
        config.optimizer.optimizer,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay,
        config.optimizer.momentum,
    )
    # Store initial learning rates for warmup scaling.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    steps_per_epoch = len(train_loader)
    scheduler, warmup_steps = get_scheduler(
        optimizer,
        config.optimizer.scheduler,
        config.epochs,
        steps_per_epoch,
        config.optimizer.warmup_epochs,
        config.optimizer.lr_min,
    )

    ema = EMAModel(model, decay=config.optimizer.ema_decay) if config.optimizer.use_ema else None
    amp_manager = AMPManager(device.type, enabled=config.use_amp)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.augmentation.label_smoothing)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    rank = get_rank()
    is_main_process = rank == 0

    if callbacks is None:
        callbacks = []
        if is_main_process:
            callbacks.append(
                CheckpointCallback(
                    config.logging.checkpoint_dir,
                    config.model.name,
                )
            )
            callbacks.append(
                EarlyStoppingCallback(
                    metric="val_acc",
                    mode="max",
                    patience=config.early_stopping_patience,
                )
            )
            callbacks.append(
                MetricsLogger(
                    config.logging.checkpoint_dir,
                    config.model.name,
                )
            )
            if config.logging.use_mlflow:
                callbacks.append(
                    MLFlowLogger(
                        config.logging.experiment_name,
                        config.logging.run_name,
                        config.to_dict(),
                    )
                )
            if config.logging.use_wandb:
                callbacks.append(
                    WandbLogger(
                        config.logging.experiment_name,
                        config.logging.run_name,
                        config.to_dict(),
                    )
                )

    logger.info(
        "Training %s for %d epochs on %s (rank %d/%d, %.2fM params)",
        config.model.name,
        config.epochs,
        device,
        rank + 1,
        get_world_size(),
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    )

    global_step = 0
    for epoch in range(config.epochs):
        if is_distributed() and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        start = time.time()

        # Warmup LR scaling for cosine scheduler.
        if config.optimizer.scheduler == "cosine" and warmup_steps > 0:
            warmup_factor = linear_warmup_lr(1.0, global_step, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * warmup_factor

        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp_manager,
            epoch,
            f"Epoch {epoch + 1}/{config.epochs} train",
            config.logging.log_interval,
            ema,
            config.optimizer.gradient_clip,
            scheduler if config.optimizer.scheduler != "cosine" else None,
        )

        eval_model = ema.ema_model if ema is not None else model
        val_loss, val_acc = evaluate_loader(eval_model, val_loader, criterion, device, amp_manager)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        epoch_metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": history["lr"][-1],
        }

        if is_main_process:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | lr=%.6f | %.1fs",
                epoch + 1,
                config.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                history["lr"][-1],
                time.time() - start,
            )
            for callback in callbacks:
                callback.on_epoch_end(epoch + 1, model, epoch_metrics)

        should_stop = any(
            isinstance(cb, EarlyStoppingCallback) and cb.should_stop for cb in callbacks
        )
        if should_stop:
            break

        global_step += steps_per_epoch

    # Restore best checkpoint on main process.
    checkpoint_dir = config.logging.checkpoint_dir
    best_path = checkpoint_dir / f"{config.model.name}_best.pt"
    if is_main_process and best_path.is_file():
        logger.info("Restoring best checkpoint: %s", best_path)
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    for callback in callbacks:
        callback.on_train_end(model)

    return history


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    """Load a model checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model
