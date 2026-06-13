"""Sanity tests for the CIFAR-10 data loader."""

from __future__ import annotations

import numpy as np
import pytest

from cifar10_eda import NUM_CLASSES, NUM_TEST, NUM_TRAIN, extract_archive, load_cifar10


@pytest.fixture(scope="module")
def cifar_dataset(tmp_path_factory):
    """Extract the archive once and load the full dataset for all tests."""
    from cifar10_eda import ARCHIVE_PATH, DATA_DIR

    extracted_dir = extract_archive(ARCHIVE_PATH, DATA_DIR)
    return load_cifar10(extracted_dir)


def test_archive_extracted(cifar_dataset):
    """Extraction should produce a non-empty dataset."""
    assert cifar_dataset.X_train.size > 0
    assert cifar_dataset.X_test.size > 0


def test_shapes(cifar_dataset):
    """Train/test tensors must match the canonical CIFAR-10 shapes."""
    assert cifar_dataset.X_train.shape == (NUM_TRAIN, 3072)
    assert cifar_dataset.y_train.shape == (NUM_TRAIN,)
    assert cifar_dataset.X_test.shape == (NUM_TEST, 3072)
    assert cifar_dataset.y_test.shape == (NUM_TEST,)


def test_dtypes(cifar_dataset):
    """Images are uint8 and labels are integers."""
    assert cifar_dataset.X_train.dtype == np.uint8
    assert cifar_dataset.X_test.dtype == np.uint8
    assert cifar_dataset.y_train.dtype == np.int64
    assert cifar_dataset.y_test.dtype == np.int64


def test_value_range(cifar_dataset):
    """Raw pixel values live in [0, 255]."""
    assert cifar_dataset.X_train.min() >= 0
    assert cifar_dataset.X_train.max() <= 255
    assert cifar_dataset.X_test.min() >= 0
    assert cifar_dataset.X_test.max() <= 255


def test_class_names(cifar_dataset):
    """There are exactly 10 human-readable class names."""
    assert len(cifar_dataset.class_names) == NUM_CLASSES
    assert all(isinstance(name, str) for name in cifar_dataset.class_names)


def test_class_balance(cifar_dataset):
    """CIFAR-10 is perfectly balanced: 5000 train and 1000 test samples per class."""
    train_counts = np.bincount(cifar_dataset.y_train, minlength=NUM_CLASSES)
    test_counts = np.bincount(cifar_dataset.y_test, minlength=NUM_CLASSES)
    assert np.all(train_counts == NUM_TRAIN // NUM_CLASSES)
    assert np.all(test_counts == NUM_TEST // NUM_CLASSES)
