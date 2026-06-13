# CIFAR-10 EDA

A production-grade exploratory data analysis project for the CIFAR-10 dataset.

## Structure

```
Mtech_Project/
├── notebooks/
│   └── CIFAR_EDA.ipynb          # Thin orchestrator: imports + runs EDA
├── src/cifar10_eda/             # Reusable package
│   ├── config.py                # Paths, seeds, constants
│   ├── data_loader.py           # Archive extraction + dataset loading
│   ├── visualization.py           # Plotting helpers
│   └── utils.py                 # Logging + reproducibility helpers
├── tests/
│   └── test_data_loader.py      # Shape/dtype/class sanity checks
├── pyproject.toml               # Package metadata + pytest config
├── requirements.txt             # Pip dependencies
└── README.md                    # This file
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Place the CIFAR-10 archive at the project root:
   ```
   Mtech_Project/cifar-10-python.tar.gz
   ```
   If it is missing, the notebook will raise a clear `FileNotFoundError`.

## Running the EDA

Start Jupyter from the project root and open `notebooks/CIFAR_EDA.ipynb`:

```bash
jupyter notebook notebooks/CIFAR_EDA.ipynb
```

The notebook now imports the `cifar10_eda` package, loads data from the local archive, and runs the visualizations.

## Running tests

```bash
pytest
```

## Design notes

- All paths are resolved relative to the project root via `pathlib`.
- Archive extraction is idempotent: it only runs if the extracted files are missing.
- Random seeds are set deterministically for reproducibility.
- Visualization helpers return matplotlib figures and support optional `save_path`.
- No Google Drive dependency remains.
