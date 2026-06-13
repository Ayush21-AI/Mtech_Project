# Production-grade refactor of CIFAR_EDA.ipynb

## Goal
Turn the single `CIFAR_EDA.ipynb` notebook into a reproducible, modular, maintainable CIFAR-10 EDA project while keeping the notebook as a thin orchestrator.

## Decisions
- **Depth:** Full modular refactor (not just in-notebook cleanup).
- **Scope:** `CIFAR_EDA.ipynb` only. `Propose_Cifar.ipynb` is left untouched unless requested later.

## Proposed structure

```
Mtech_Project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── CIFAR_EDA.ipynb          # thin orchestrator: imports + runs EDA
├── src/
│   └── cifar10_eda/
│       ├── __init__.py
│       ├── config.py            # paths, seeds, constants
│       ├── data_loader.py       # archive extraction + dataset loading
│       ├── visualization.py     # plotting helpers
│       └── utils.py             # logging + seed setup
└── tests/
    └── test_data_loader.py      # shape/dtype/class sanity checks
```

## Implementation steps

1. **Fix the notebook typo** in `CIFAR_EDA.ipynb` cell-0 (`th #` → clean comment).
2. **Create `src/cifar10_eda/config.py`**
   - Resolve all paths via `pathlib` from `__file__`.
   - Default archive: `Mtech_Project/cifar-10-python.tar.gz`.
   - Default extracted dir: `Mtech_Project/data/cifar-10-batches-py`.
   - Expose `RANDOM_SEED`, `CLASS_NAMES`, `IMAGE_SHAPE`, etc.
3. **Create `src/cifar10_eda/utils.py`**
   - `setup_logging()` with consistent format.
   - `set_seed(seed)` for NumPy/TensorFlow random state.
4. **Create `src/cifar10_eda/data_loader.py`**
   - `extract_archive(archive_path, extract_to, force=False)` — idempotent extraction using `with tarfile.open(...)`.
   - `load_cifar10(data_dir)` — loads 5 train batches, test batch, and meta; returns a dataclass/namedtuple with `X_train`, `y_train`, `X_test`, `y_test`, `class_names`.
   - Add type hints, docstrings, and checksum verification for the archive (optional, known CIFAR-10 SHA256).
5. **Create `src/cifar10_eda/visualization.py`**
   - Move `show_unique_images`, `show_images`, `drop_color_channel`, `display_color_hists` here.
   - Add `plot_class_distribution(train_labels, test_labels, class_names)`.
   - Make functions accept `save_path` and return `Figure` objects for notebook reuse.
   - Remove mutable default arguments and use `None` defaults.
6. **Rewrite `notebooks/CIFAR_EDA.ipynb`**
   - Keep markdown explanations.
   - Cells do only:
     - `%autoreload 2` + import src modules
     - setup logging/seeds
     - load data
     - run visualizations
     - print dataset sanity report
7. **Add `requirements.txt`** with minimal EDA dependencies.
8. **Add `tests/test_data_loader.py`** using `pytest` or `unittest`.
   - Assert shapes `(50000, 3072)`, `(10000, 3072)`.
   - Assert 10 classes, 5000/1000 per class.
   - Assert value range `0–255`, dtype `uint8`.
9. **Add `README.md`** explaining setup, run instructions, and data source.

## Trade-offs
- **Package vs. notebook-only:** A package is more reusable/testable; notebook stays readable.
- **Relative paths vs. env vars:** Use project-root relative paths by default. Env vars can override if needed.
- **Checksum verification:** Optional, because the local archive is trusted. Can be added later.

## Success criteria
- Running the notebook end-to-end loads local data without Google Drive.
- All functions are importable and unit-testable.
- No hardcoded absolute paths outside the project.
- Clear logging replaces print statements.
- `pytest tests/` passes.
