"""Estimate target-class mask anomalies from a clean image set."""

import argparse
from collections import Counter
import json
from pathlib import Path

from triggercraft import scan_imagefolder
from triggercraft.data import balanced_indices
from triggercraft.defenses import NeuralCleanseDetector
from triggercraft.models import load_classifier
from triggercraft.utils import parse_experiment_args, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--backend", choices=["timm", "torchvision"], default="timm")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "neural_cleanse")
    if args.samples < 1:
        parser.error("samples must be positive")
    import torch
    from torchvision.datasets import ImageFolder

    seed_everything(args.seed)
    _, classes = scan_imagefolder(args.clean)
    classifier = load_classifier(args.model, len(classes), backend=args.backend,
                                 checkpoint=args.checkpoint, input_size=args.input_size,
                                 device=args.device)
    dataset = ImageFolder(args.clean, transform=classifier.val_transform)
    if dataset.class_to_idx != classes:
        raise ValueError("Class order changed while loading the clean dataset")
    indices = balanced_indices(dataset.targets, args.samples, args.seed)
    count = len(indices)
    images = torch.stack([dataset[index][0] for index in indices])
    detector = NeuralCleanseDetector(classifier.model, steps=args.steps, lr=args.lr,
                                     regularization=args.regularization, device=args.device)
    sample_counts = Counter(dataset.targets[index] for index in indices)
    result = {"method": "neural_cleanse", "model": args.model, "backend": args.backend,
              "class_to_idx": classes,
              "sample_class_counts": {name: sample_counts[class_index]
                                      for name, class_index in classes.items()},
              **detector.scan(images, len(classes))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Scanned {len(classes)} target classes with {count} images")


if __name__ == "__main__":
    main()
