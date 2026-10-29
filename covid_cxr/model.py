"""ResNet-18 with a single COVID-vs-Normal output."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from covid_cxr.config import CLASS_TO_INDEX, DISCLAIMER, IMAGENET_MEAN, IMAGENET_STD, POSITIVE_CLASS
from covid_cxr.preprocessing import preprocessing_config


def assert_imagenet_normalization() -> None:
    """Check mean and standard deviation against the pretrained weight recipe.

    The default ResNet-18 preprocessing also center-crops. This project does not
    use that crop. It only reuses the normalization statistics.
    """
    recipe = ResNet18_Weights.IMAGENET1K_V1.transforms()
    mean = tuple(float(value) for value in recipe.mean)
    std = tuple(float(value) for value in recipe.std)
    if mean != IMAGENET_MEAN or std != IMAGENET_STD:
        raise RuntimeError(
            "The ImageNet normalization constants in this project do not match "
            f"ResNet-18 IMAGENET1K_V1 (mean={mean}, std={std})."
        )


def build_model(pretrained: bool = True) -> nn.Module:
    if pretrained:
        assert_imagenet_normalization()
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def set_phase(model: nn.Module, phase: str) -> None:
    """Train the new head, or the head plus the final residual block."""
    if phase not in {"head", "layer4"}:
        raise ValueError(f"Unknown training phase: {phase}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if phase == "layer4":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def trainable_batchnorm_in_train_mode(model: nn.Module) -> None:
    """Keep frozen batch-norm layers from updating their running statistics."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def cpu_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(path: str | Path, model: nn.Module, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "resnet18",
        "pretrained_weights": "IMAGENET1K_V1",
        "num_outputs": 1,
        "positive_class": POSITIVE_CLASS,
        "class_to_index": CLASS_TO_INDEX,
        "model_state_dict": cpu_state_dict(model),
        "preprocessing": preprocessing_config(),
        "threshold": None,
        "threshold_source": None,
        "disclaimer": DISCLAIMER,
    }
    checkpoint.update(payload)
    checkpoint["model_state_dict"] = cpu_state_dict(model)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=map_location)
    if checkpoint.get("architecture") != "resnet18" or checkpoint.get("num_outputs") != 1:
        raise RuntimeError("This checkpoint is not a binary ResNet-18 saved by this project.")
    if checkpoint.get("positive_class") != POSITIVE_CLASS:
        raise RuntimeError("Checkpoint positive class is not COVID.")
    return checkpoint


def model_from_checkpoint(checkpoint: dict) -> nn.Module:
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
