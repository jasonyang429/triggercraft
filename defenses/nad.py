"""Repair a classifier using clean-data Neural Attention Distillation."""

import argparse
import json
from pathlib import Path

from triggercraft.data import balanced_indices
from triggercraft.models import load_classifier
from triggercraft.nad import AttentionDistiller
from triggercraft.utils import parse_experiment_args, seed_everything
from triggercraft.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Root with train/ and val/ class folders")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--backend", choices=["timm", "torchvision"], default="timm")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", default="layer2,layer3,layer4")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--teacher-epochs", type=int, default=1)
    parser.add_argument("--student-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "nad")

    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    seed_everything(args.seed)
    train_root, val_root = args.data / "train", args.data / "val"
    classes = ImageFolder(train_root).classes
    classifier = load_classifier(args.model, len(classes), backend=args.backend,
                                 checkpoint=args.checkpoint, input_size=args.input_size,
                                 device=args.device)
    train_data = ImageFolder(train_root, transform=classifier.train_transform)
    val_data = ImageFolder(val_root, transform=classifier.val_transform)
    if train_data.classes != val_data.classes:
        raise ValueError("train and val must have the same classes")
    indices = balanced_indices(train_data.targets, args.samples, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(train_data, indices), batch_size=args.batch_size,
                              shuffle=True, generator=generator, num_workers=args.workers)
    before = evaluate(classifier.model, val_data, args.batch_size, args.workers, args.device)
    layers = [name.strip() for name in args.layers.split(",") if name.strip()]
    distiller = AttentionDistiller(classifier.model, layers, beta=args.beta,
                                   power=args.power, device=args.device)
    history = distiller.fit(train_loader, teacher_epochs=args.teacher_epochs,
                            student_epochs=args.student_epochs, lr=args.lr)
    after = evaluate(classifier.model, val_data, args.batch_size, args.workers, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.model.state_dict(), args.output / "repaired_model.pth")
    report = {"method": "nad", "model": args.model, "backend": args.backend,
              "layers": layers, "train_samples": len(indices), "classes": classes,
              "teacher_epochs": args.teacher_epochs, "student_epochs": args.student_epochs,
              "beta": args.beta, "history": history, "before": before, "after": after}
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"clean accuracy: {before['accuracy']:.4f} -> {after['accuracy']:.4f}")


if __name__ == "__main__":
    main()
