"""Metrics tracking and evaluation helpers."""

from __future__ import annotations

import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader


class MetricsTracker:
    """Simple accumulator for loss and accuracy over batches."""

    def __init__(self, num_classes: int = 10) -> None:
        self.num_classes = num_classes
        self.total_loss = 0.0
        self.total_samples = 0
        self.correct = 0

    def update(
        self,
        loss: float,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        _, predicted = outputs.max(1)
        if targets.ndim == 1:
            self.correct += predicted.eq(targets).sum().item()
        else:
            # Soft targets: take argmax of one-hot/soft labels.
            hard_targets = targets.argmax(dim=1)
            self.correct += predicted.eq(hard_targets).sum().item()

    def avg_loss(self) -> float:
        return self.total_loss / max(self.total_samples, 1)

    def accuracy(self) -> float:
        return self.correct / max(self.total_samples, 1)


def build_classification_metrics(num_classes: int, device: torch.device) -> dict[str, torchmetrics.Metric]:
    """Create a dict of torchmetrics for evaluation."""
    return {
        "accuracy": torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, average="micro"
        ).to(device),
        "top5": torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, top_k=5, average="micro"
        ).to(device),
        "f1": torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes, average="macro"
        ).to(device),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_manager=None,
) -> tuple[float, float]:
    """Evaluate a model on a data loader.

    Returns
    -------
    tuple[float, float]
        ``(average_loss, accuracy)``.
    """
    model.eval()
    tracker = MetricsTracker()

    use_amp = amp_manager is not None and amp_manager.enabled

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if use_amp:
                with amp_manager.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            tracker.update(loss.item(), outputs, targets, inputs.size(0))

    return tracker.avg_loss(), tracker.accuracy()
