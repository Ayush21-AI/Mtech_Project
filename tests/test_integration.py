"""Integration test that exercises the real default config end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest

from cifar10_models import fit
from cifar10_models.config import TrainConfig
from cifar10_models.data import get_dataloaders
from cifar10_models.evaluate import evaluate
from cifar10_models.models.model_factory import build_model
from cifar10_models.optim import AMPManager
from cifar10_models.utils import load_config


def test_default_config_runs_two_epochs() -> None:
    """The default config (pin_memory=True, use_ema=True, cosine) must run and
    the learning rate must change across epochs, showing that warmup + cosine
    are actually being stepped.
    """
    config = load_config(Path("configs/default.yaml"))
    config.data.fast_dev_run = True
    config.data.fast_dev_size = 500
    config.data.num_workers = 0
    # Keep pin_memory=True (the default): this is the exact path that previously
    # crashed with a NameError inside get_dataloaders, so the regression test
    # must actually exercise it rather than short-circuiting around it.
    config.data.pin_memory = True
    config.epochs = 2
    config.use_amp = False
    config.compile_model = False

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        augmentation_cfg=config.augmentation,
        data_cfg=config.data,
        num_classes=config.model.num_classes,
    )
    model = build_model(config.model)

    history, eval_model = fit(model, train_loader, val_loader, config)

    assert len(history["train_loss"]) == 2
    assert len(history["val_acc"]) == 2
    assert len(history["lr"]) == 2
    # LR must actually change to prove the scheduler is being stepped.
    assert history["lr"][0] != history["lr"][1]

    amp_manager = AMPManager(config.device.type, enabled=False)
    test_loss, test_acc = evaluate(eval_model, test_loader, config.device, amp_manager)
    assert 0 <= test_acc <= 1
