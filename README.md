# CIFAR-10 Image Classifier

A clean PyTorch project that trains and compares different image classification models on the CIFAR-10 dataset.

## What this project does

- Loads CIFAR-10 images from a local `cifar-10-python.tar.gz` file.
- Trains one of four models: **ConvMixer**, **Patch CNN**, **Vision Transformer**, or **ResNet-18**.
- Uses modern training tricks: AdamW, warmup + cosine LR, data augmentation (RandAugment, CutMix, Mixup), AMP, EMA, early stopping, and checkpointing.
- Evaluates the trained model on the test set.
- Optionally exports the trained model to ONNX format for deployment.
- Keeps everything modular and config-driven.

## Project structure

```
Mtech_Project/
├── configs/                    # YAML training configs
│   ├── default.yaml            # ConvMixer config
│   ├── patch_cnn.yaml
│   ├── vit.yaml
│   └── resnet18.yaml
├── notebooks/
│   ├── CIFAR_EDA.ipynb         # EDA notebook
│   └── CIFAR_Models.ipynb      # Training notebook
├── src/
│   ├── cifar10_eda/            # EDA package
│   └── cifar10_models/         # PyTorch training package
├── tests/                      # Pytest tests
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Put the CIFAR-10 archive in the project root:
   ```
   Mtech_Project/cifar-10-python.tar.gz
   ```
   If you don't have it, download it from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

## How to run

### 1. Run tests to check everything works

```bash
pytest
```

You should see all tests pass.

### 2. Train a model from the command line

```bash
python -m cifar10_models --config configs/default.yaml
```

This trains ConvMixer on the full CIFAR-10 dataset for 50 epochs and saves the best checkpoint to `checkpoints/`.

To train a different model:

```bash
python -m cifar10_models --config configs/vit.yaml
```

### 3. Quick smoke test (fast training on a small subset)

```bash
python -m cifar10_models \
  --config configs/default.yaml \
  --override data.fast_dev_run=true \
  --override data.fast_dev_size=500 \
  --override epochs=1 \
  --override data.num_workers=0 \
  --export-onnx \
  --test-tta
```

This:
- Trains on only 500 images for 1 epoch.
- Evaluates on the test set.
- Runs test-time augmentation.
- Exports the model to `checkpoints/convmixer.onnx`.

### 4. Train in Jupyter notebook

```bash
jupyter notebook notebooks/CIFAR_Models.ipynb
```

### 5. Override any config from the command line

```bash
python -m cifar10_models \
  --config configs/default.yaml \
  --override epochs=10 \
  --override data.batch_size=64
```

## Available models

| Model         | Config file              |
|---------------|--------------------------|
| ConvMixer     | `configs/default.yaml`   |
| Patch CNN     | `configs/patch_cnn.yaml` |
| Vision Transformer | `configs/vit.yaml`  |
| ResNet-18     | `configs/resnet18.yaml`  |

Switch models by changing `--config`.

## Outputs

- **Checkpoints**: `checkpoints/<model>_best.pt` and `checkpoints/<model>_last.pt`
- **ONNX model**: `checkpoints/<model>.onnx` (if `--export-onnx` is used)
- **Metrics log**: `checkpoints/<model>_metrics.jsonl`

## Features included

- Config-driven training via YAML files.
- Multiple model architectures in one place.
- Modern training recipe: AdamW, warmup + cosine LR, gradient clipping, AMP, EMA.
- Data augmentation: RandAugment, CutMix, Mixup, RandomErasing.
- Best checkpointing, early stopping, and training metrics logging.
- Test-time augmentation during evaluation.
- ONNX export with runtime parity check.
- Distributed training scaffolding: process-group setup, DistributedSampler, and DDP wrapping are wired in for multi-GPU runs.

## Notes

- The project uses MPS if available (Apple Silicon), otherwise CUDA, then CPU.
- For real training, use the full dataset and run for at least 50 epochs.
- `fast_dev_run=true` is only for quick smoke tests.

## Repository

GitHub: https://github.com/Ayush21-AI/ML_MTech_Project
