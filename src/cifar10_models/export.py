"""ONNX export utilities for CIFAR-10 models."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from cifar10_models.augmentation import CIFAR_MEAN, CIFAR_STD

logger = logging.getLogger("cifar10_models")


def _sanitize_batchnorm(model: nn.Module) -> None:
    """Reset BatchNorm running stats if they contain NaN values."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if torch.isnan(module.running_mean).any() or torch.isnan(module.running_var).any():
                logger.warning("BatchNorm running stats contain NaN; resetting for export.")
                module.reset_running_stats()


class NormalizeWrapper(nn.Module):
    """Wrap a model with baked-in CIFAR-10 normalization."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)


def export_to_onnx(
    model: nn.Module,
    export_path: Path,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """Export a CIFAR-10 model to ONNX with a dynamic batch axis.

    Parameters
    ----------
    model: nn.Module
        Trained model.
    export_path: Path
        Output ONNX path.
    opset_version: int
        ONNX opset version.
    dynamic_batch: bool
        Whether to mark the batch dimension dynamic.
    verify: bool
        Whether to run an ONNX Runtime parity check.

    Returns
    -------
    Path
        Path to the exported ONNX file.
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize BatchNorm running stats: tiny fast-dev runs or unstable
    # training can leave NaN values, which break torch.export. Reset them
    # before wrapping so the exported graph is clean.
    _sanitize_batchnorm(model)

    wrapped = NormalizeWrapper(model)
    wrapped.eval()
    # torch.export-based ONNX exporter is not device-agnostic on MPS; use CPU.
    wrapped = wrapped.cpu()

    # Use a non-unity batch dummy input when dynamic batch is requested.
    # Some transformer patterns (class token expand + pos_embed broadcast)
    # are easier for torch.export to keep dynamic with batch_size > 1.
    dummy_batch = 2 if dynamic_batch else 1
    dummy_input = torch.randn(dummy_batch, 3, 32, 32)
    dynamic_shapes = ({0: "batch"},) if dynamic_batch else None

    onnx_program = torch.onnx.export(
        wrapped,
        (dummy_input,),
        export_path,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_shapes=dynamic_shapes,
    )

    if onnx_program is not None:
        onnx_program.save(str(export_path))

    # Re-load the saved model and run the parity check on the saved file.
    # This catches any issues introduced by version conversion or serialization.
    onnx.checker.check_model(str(export_path))
    logger.info("ONNX model exported to %s", export_path)

    if verify:
        _verify_onnx_runtime(wrapped, export_path)

    return export_path


def _verify_onnx_runtime(model: nn.Module, onnx_path: Path) -> None:
    """Compare PyTorch and ONNX Runtime outputs on multiple batch sizes."""
    model.eval()
    session = ort.InferenceSession(str(onnx_path))

    for batch_size in (1, 4):
        dummy_input = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            pt_out = model(dummy_input).numpy()

        onnx_out = session.run(None, {"input": dummy_input.numpy()})[0]
        np.testing.assert_allclose(
            pt_out,
            onnx_out,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"ONNX parity failed for batch_size={batch_size}",
        )

    logger.info("ONNX Runtime parity check passed")
