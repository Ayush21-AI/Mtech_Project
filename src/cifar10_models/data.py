"""CIFAR-10 PyTorch data loading and splitting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose

from cifar10_eda import ARCHIVE_PATH, DATA_DIR, NUM_CLASSES, extract_archive, load_cifar10
from cifar10_models.augmentation import (
    CutMixMixupCollator,
    build_test_transforms,
    build_train_transforms,
)
from cifar10_models.config import get_device
from cifar10_models.distributed import get_distributed_sampler

logger = logging.getLogger("cifar10_models")


class Cifar10Dataset(Dataset):
    """PyTorch Dataset wrapping the CIFAR-10 flat arrays.

    Raw data is ``(N, 3072)`` uint8 in channel-first CIFAR layout
    (red plane, green plane, blue plane). This class reshapes it to
    ``(N, 3, 32, 32)`` PIL-compatible images and applies a torchvision
    transform.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Compose | None = None,
    ) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same length")

        self.images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            from PIL import Image

            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)


def extract_and_load(
    archive_path: Path = ARCHIVE_PATH,
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract the local archive and return CIFAR-10 arrays."""
    extracted_dir = extract_archive(archive_path, data_dir)
    dataset = load_cifar10(extracted_dir)
    return (
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        dataset.class_names,
    )


def _split_indices(
    total: int,
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic train/validation indices."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator)
    val_size = int(val_split * total)
    train_indices = perm[val_size:].tolist()
    val_indices = perm[:val_size].tolist()
    return train_indices, val_indices


def get_dataloaders(
    augmentation_cfg,
    data_cfg,
    num_classes: int = NUM_CLASSES,
    archive_path: Path = ARCHIVE_PATH,
    data_dir: Path = DATA_DIR,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/validation/test DataLoaders from the local archive.

    Parameters
    ----------
    augmentation_cfg: AugmentationConfig
        Augmentation settings.
    data_cfg: DataConfig
        DataLoader settings.
    num_classes: int
        Number of output classes.
    archive_path: Path
        Path to ``cifar-10-python.tar.gz``.
    data_dir: Path
        Directory where the archive will be extracted if needed.
    distributed: bool
        Whether this is a DDP process group.
    rank: int
        Current DDP rank.
    world_size: int
        Total number of DDP processes.

    Returns
    -------
    tuple
        ``train_loader, val_loader, test_loader, class_names``.
    """
    X_train, y_train, X_test, y_test, class_names = extract_and_load(
        archive_path, data_dir
    )

    train_transform = build_train_transforms(augmentation_cfg)
    test_transform = build_test_transforms()

    full_train_dataset = Cifar10Dataset(X_train, y_train, transform=train_transform)
    val_dataset = Cifar10Dataset(X_train, y_train, transform=test_transform)
    test_dataset = Cifar10Dataset(X_test, y_test, transform=test_transform)

    if data_cfg.fast_dev_run:
        train_indices = list(range(min(data_cfg.fast_dev_size, len(full_train_dataset))))
        val_indices = list(range(min(data_cfg.fast_dev_size // 5, len(val_dataset))))
        test_indices = list(range(min(data_cfg.fast_dev_size // 5, len(test_dataset))))
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        test_subset = Subset(test_dataset, test_indices)
    else:
        train_indices, val_indices = _split_indices(
            len(full_train_dataset),
            data_cfg.validation_split,
            data_cfg.seed,
        )
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        test_subset = test_dataset

    train_sampler = get_distributed_sampler(
        train_subset,
        shuffle=True,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=data_cfg.seed,
    )
    # Validation/test are intentionally NOT sharded across DDP ranks: each rank
    # evaluates the full set so the metrics that drive checkpointing and early
    # stopping (and the final reported accuracy) are computed over every sample,
    # not a 1/world_size slice. Only training is data-parallel.
    val_sampler = None
    test_sampler = None

    # Disable CutMix/Mixup for tiny fast-dev subsets to keep smoke tests stable.
    use_cutmix_mixup = (
        augmentation_cfg.cutmix or augmentation_cfg.mixup
    ) and not data_cfg.fast_dev_run
    cutmix_collator = CutMixMixupCollator(
        cutmix_alpha=augmentation_cfg.cutmix_alpha,
        mixup_alpha=augmentation_cfg.mixup_alpha,
        num_classes=num_classes,
        cutmix=augmentation_cfg.cutmix and use_cutmix_mixup,
        mixup=augmentation_cfg.mixup and use_cutmix_mixup,
    )

    # pin_memory only helps on CUDA; on MPS/CPU it can warn or slow things down.
    should_pin = data_cfg.pin_memory and get_device().type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=should_pin,
        drop_last=not data_cfg.fast_dev_run,
        collate_fn=cutmix_collator,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=should_pin,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=data_cfg.batch_size,
        sampler=test_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=should_pin,
        drop_last=False,
    )

    logger.info(
        "DataLoaders ready: train=%d, val=%d, test=%d, classes=%d",
        len(train_subset),
        len(val_subset),
        len(test_subset),
        len(class_names),
    )
    return train_loader, val_loader, test_loader, class_names
