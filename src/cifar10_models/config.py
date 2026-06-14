"""Config-driven training configuration for CIFAR-10 models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from cifar10_eda import NUM_CLASSES


def get_device() -> torch.device:
    """Select the best available device: MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""

    random_crop: bool = True
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 12
    color_jitter: float = 0.1
    random_erasing: float = 0.25
    cutmix: bool = True
    cutmix_alpha: float = 1.0
    mixup: bool = True
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.1


@dataclass
class DataConfig:
    """Dataset and dataloader settings."""

    batch_size: int = 128
    num_workers: int = 4
    validation_split: float = 0.1
    pin_memory: bool = True
    fast_dev_run: bool = False
    fast_dev_size: int = 500
    seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture settings."""

    name: str = "convmixer"
    patch_size: int = 4
    embed_dim: int = 256
    depth: int = 8
    kernel_size: int = 5
    num_heads: int = 4
    mlp_dim: int = 512
    dropout: float = 0.1
    stochastic_depth: float = 0.0
    use_cnn_stem: bool = False
    num_classes: int = NUM_CLASSES

    def __post_init__(self) -> None:
        allowed = {"patch_cnn", "convmixer", "vit", "resnet18"}
        if self.name not in allowed:
            raise ValueError(f"model.name must be one of {allowed}, got {self.name}")


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler settings."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    lr_min: float = 1e-6
    gradient_clip: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.9999

    def __post_init__(self) -> None:
        allowed_opt = {"adamw", "sgd", "adam"}
        if self.optimizer not in allowed_opt:
            raise ValueError(f"optimizer must be one of {allowed_opt}, got {self.optimizer}")
        allowed_sched = {"cosine", "onecycle", "step", "none"}
        if self.scheduler not in allowed_sched:
            raise ValueError(f"scheduler must be one of {allowed_sched}, got {self.scheduler}")


@dataclass
class LoggingConfig:
    """Experiment tracking and checkpointing settings."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "checkpoints")
    log_interval: int = 50
    use_mlflow: bool = False
    use_wandb: bool = False
    experiment_name: str = "cifar10-models"
    run_name: str | None = None


@dataclass
class ExportConfig:
    """ONNX export settings."""

    opset_version: int = 18
    dynamic_batch: bool = True


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    epochs: int = 50
    use_amp: bool = True
    compile_model: bool = False
    early_stopping_patience: int = 10
    test_time_augmentation: bool = False
    seed: int = 42
    device: torch.device = field(default_factory=get_device)

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a plain nested dict for logging."""

        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, torch.device):
                return str(obj)
            if isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in obj.__dict__.items()}
            return obj

        return _convert(self.__dict__)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        """Build a TrainConfig from a nested dict (e.g. loaded YAML)."""
        raw = raw.copy()
        model = ModelConfig(**raw.pop("model", {}))
        data = DataConfig(**raw.pop("data", {}))
        augmentation = AugmentationConfig(**raw.pop("augmentation", {}))
        optimizer = OptimizerConfig(**raw.pop("optimizer", {}))
        logging_cfg = LoggingConfig(**raw.pop("logging", {}))
        export = ExportConfig(**raw.pop("export", {}))
        return cls(
            model=model,
            data=data,
            augmentation=augmentation,
            optimizer=optimizer,
            logging=logging_cfg,
            export=export,
            **raw,
        )
