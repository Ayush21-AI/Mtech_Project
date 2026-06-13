"""CIFAR-10 exploratory data analysis package."""

from cifar10_eda.config import (
    ARCHIVE_PATH,
    DATA_DIR,
    EXTRACTED_DIR,
    IMAGE_SHAPE,
    NUM_CLASSES,
    NUM_TEST,
    NUM_TRAIN,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from cifar10_eda.data_loader import Cifar10Dataset, extract_archive, load_cifar10
from cifar10_eda.utils import set_seed, setup_logging
from cifar10_eda.visualization import (
    display_color_hists,
    drop_color_channel,
    plot_class_distribution,
    show_images,
    show_unique_images,
)

__all__ = [
    "ARCHIVE_PATH",
    "DATA_DIR",
    "EXTRACTED_DIR",
    "IMAGE_SHAPE",
    "NUM_CLASSES",
    "NUM_TEST",
    "NUM_TRAIN",
    "PROJECT_ROOT",
    "RANDOM_SEED",
    "Cifar10Dataset",
    "extract_archive",
    "load_cifar10",
    "set_seed",
    "setup_logging",
    "display_color_hists",
    "drop_color_channel",
    "plot_class_distribution",
    "show_images",
    "show_unique_images",
]
