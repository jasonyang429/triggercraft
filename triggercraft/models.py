"""Model construction and its paired preprocessing live in one place."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class ImageClassifier:
    model: Any
    train_transform: Callable
    val_transform: Callable
    class_count: int
    name: str
    backend: str


def load_classifier(name: str, class_count: int, *, backend: str = "timm",
                    checkpoint: str | Path | None = None, input_size: int = 224,
                    pretrained: bool = False, device: str = "cpu") -> ImageClassifier:
    """Create a timm or torchvision classifier and compatible image transforms."""
    import torch
    from torchvision import transforms

    if class_count < 1 or input_size < 32:
        raise ValueError("class_count must be positive and input_size at least 32")
    if backend == "timm":
        import timm
        from timm.data import create_transform, resolve_model_data_config

        model = timm.create_model(name, pretrained=pretrained, num_classes=class_count)
        data_config = resolve_model_data_config(model)
        data_config["input_size"] = (3, input_size, input_size)
        train_transform = create_transform(**data_config, is_training=True)
        val_transform = create_transform(**data_config, is_training=False)
    elif backend == "torchvision":
        from torchvision.models import get_model

        if pretrained:
            raise ValueError("torchvision pretrained weights with a custom class count are unsupported; use the timm backend")
        model = get_model(name, weights=None, num_classes=class_count)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std)])
        val_transform = transforms.Compose([
            transforms.Resize(round(input_size * 256 / 224)),
            transforms.CenterCrop(input_size), transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'timm' or 'torchvision'")

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state)
    return ImageClassifier(model.to(device), train_transform, val_transform,
                           class_count, name, backend)


def load_timm_model(name: str, class_count: int, checkpoint: str | Path | None = None,
                    input_size: int = 224, pretrained: bool = False,
                    device: str = "cpu") -> ImageClassifier:
    """Compatibility shortcut for existing timm experiments."""
    return load_classifier(name, class_count, backend="timm", checkpoint=checkpoint,
                           input_size=input_size, pretrained=pretrained, device=device)
