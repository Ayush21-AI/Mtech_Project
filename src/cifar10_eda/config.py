"""Project-wide configuration and constants."""

from pathlib import Path

# Resolve the project root as the directory that contains `src/`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_PATH = PROJECT_ROOT / "cifar-10-python.tar.gz"
EXTRACTED_DIR = DATA_DIR / "cifar-10-batches-py"

RANDOM_SEED = 42

# CIFAR-10 known shape and size constants.
IMAGE_SHAPE = (32, 32, 3)
FLAT_DIM = 3072
NUM_CLASSES = 10
NUM_TRAIN = 50000
NUM_TEST = 10000
TRAIN_BATCHES = 5

# Expected files inside the extracted archive.
TRAIN_BATCH_FILENAMES = [f"data_batch_{i}" for i in range(1, TRAIN_BATCHES + 1)]
TEST_BATCH_FILENAME = "test_batch"
META_FILENAME = "batches.meta"
