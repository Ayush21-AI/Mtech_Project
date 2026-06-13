"""CIFAR-10 dataset loading and archive extraction."""

from __future__ import annotations

import logging
import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cifar10_eda.config import (
    ARCHIVE_PATH,
    DATA_DIR,
    EXTRACTED_DIR,
    FLAT_DIM,
    IMAGE_SHAPE,
    META_FILENAME,
    NUM_CLASSES,
    NUM_TEST,
    NUM_TRAIN,
    TEST_BATCH_FILENAME,
    TRAIN_BATCHES,
    TRAIN_BATCH_FILENAMES,
)

logger = logging.getLogger("cifar10_eda")


@dataclass(frozen=True)
class Cifar10Dataset:
    """Container for the CIFAR-10 dataset."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    image_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        # Basic invariants supplied by the loader; kept here as documentation.
        object.__setattr__(self, "_shape_ok", True)


def _decode_bytes(obj: bytes | list[bytes] | dict[bytes, Any]) -> str | list[str] | dict[str, Any]:
    """Recursively decode CIFAR-10's legacy ``bytes`` keys/values to ``str``."""
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    if isinstance(obj, list):
        return [_decode_bytes(item) for item in obj]
    if isinstance(obj, dict):
        return {_decode_bytes(k): _decode_bytes(v) for k, v in obj.items()}
    return obj


def _unpickle(file_path: Path) -> dict:
    """Unpickle a CIFAR-10 batch file and decode legacy byte strings."""
    with open(file_path, "rb") as file:
        raw = pickle.load(file, encoding="bytes")
    return _decode_bytes(raw)


def _is_extracted(extracted_dir: Path) -> bool:
    """Check whether the archive has already been extracted."""
    expected_files = [*TRAIN_BATCH_FILENAMES, TEST_BATCH_FILENAME, META_FILENAME]
    if not extracted_dir.is_dir():
        return False
    return all((extracted_dir / name).is_file() for name in expected_files)


def extract_archive(
    archive_path: Path = ARCHIVE_PATH,
    extract_to: Path = DATA_DIR,
    force: bool = False,
) -> Path:
    """Extract the CIFAR-10 ``.tar.gz`` archive exactly once.

    Parameters
    ----------
    archive_path: Path
        Path to ``cifar-10-python.tar.gz``.
    extract_to: Path
        Directory where the ``cifar-10-batches-py`` folder will be created.
    force: bool
        If ``True``, re-extract even if the files already exist.

    Returns
    -------
    Path
        Path to the extracted ``cifar-10-batches-py`` directory.
    """
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)

    if not archive_path.is_file():
        raise FileNotFoundError(f"CIFAR-10 archive not found: {archive_path}")

    extracted_dir = extract_to / "cifar-10-batches-py"

    if not force and _is_extracted(extracted_dir):
        logger.info("CIFAR-10 archive already extracted at %s", extracted_dir)
        return extracted_dir

    logger.info("Extracting %s to %s", archive_path, extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    logger.info("Extraction complete")

    if not _is_extracted(extracted_dir):
        raise RuntimeError("Archive extracted but expected CIFAR-10 files are missing")

    return extracted_dir


def load_cifar10(extracted_dir: Path = EXTRACTED_DIR) -> Cifar10Dataset:
    """Load CIFAR-10 train/test images, labels, and class names.

    Parameters
    ----------
    extracted_dir: Path
        Directory containing ``data_batch_*``, ``test_batch``, and ``batches.meta``.

    Returns
    -------
    Cifar10Dataset
        Structured dataset with train/test splits and human-readable class names.
    """
    extracted_dir = Path(extracted_dir)
    if not _is_extracted(extracted_dir):
        raise FileNotFoundError(
            f"CIFAR-10 data directory is incomplete: {extracted_dir}. "
            "Run extract_archive() first."
        )

    train_images: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    for batch_name in TRAIN_BATCH_FILENAMES:
        batch = _unpickle(extracted_dir / batch_name)
        train_images.append(np.asarray(batch["data"], dtype=np.uint8))
        train_labels.append(np.asarray(batch["labels"], dtype=np.int64))

    X_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    test_batch = _unpickle(extracted_dir / TEST_BATCH_FILENAME)
    X_test = np.asarray(test_batch["data"], dtype=np.uint8)
    y_test = np.asarray(test_batch["labels"], dtype=np.int64)

    meta = _unpickle(extracted_dir / META_FILENAME)
    class_names = list(meta["label_names"])

    # Sanity checks.
    if X_train.shape != (NUM_TRAIN, FLAT_DIM):
        raise ValueError(f"Unexpected train shape: {X_train.shape}")
    if X_test.shape != (NUM_TEST, FLAT_DIM):
        raise ValueError(f"Unexpected test shape: {X_test.shape}")
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, got {len(class_names)}")

    logger.info("Loaded CIFAR-10: train=%s, test=%s, classes=%s", X_train.shape, X_test.shape, class_names)

    return Cifar10Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        image_shape=IMAGE_SHAPE,
    )
