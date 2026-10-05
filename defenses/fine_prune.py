"""Measure clean performance while masking low-activation feature channels."""

import argparse
import json
from pathlib import Path

from triggercraft import ImageFiles, scan_imagefolder
from triggercraft.data import balanced_indices
from triggercraft.defenses import FinePruner
from triggercraft.models import load_classifier
from triggercraft.utils import parse_experiment_args, seed_everything
from triggercraft.evaluation import evaluate, evaluate_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--query", type=Path)
    parser.add_argument("--target-class")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--backend", choices=["timm", "torchvision"], default="timm")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", required=True, help="Feature layer from model.named_modules()")
    parser.add_argument("--fractions", default="0,0.1,0.25,0.5")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "fine_prune")

    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset

    seed_everything(args.seed)
    _, classes = scan_imagefolder(args.clean)
    classifier = load_classifier(args.model, len(classes), backend=args.backend,
                                 checkpoint=args.checkpoint, input_size=args.input_size,
                                 device=args.device)
    dataset = ImageFolder(args.clean, transform=classifier.val_transform)
    if dataset.class_to_idx != classes:
        raise ValueError("Class order changed while loading clean data")
    indices = balanced_indices(dataset.targets, args.samples, args.seed)
    rank_loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                             num_workers=args.workers)
    query = ImageFiles(args.query, classifier.val_transform) if args.query else None
    target_index = None
    if args.target_class is not None:
        target_index = classes.get(str(args.target_class))
        if target_index is None and str(args.target_class).isdigit():
            target_index = int(args.target_class)
        if target_index is None or not 0 <= target_index < len(classes):
            parser.error("target-class must name a clean class or its numeric index")
    try:
        fractions = [float(item.strip()) for item in args.fractions.split(",")]
    except ValueError:
        parser.error("fractions must be comma-separated numbers")
    pruner = FinePruner(classifier.model, args.layer, args.device)
    ranking = pruner.rank(rank_loader)
    rows = []
    try:
        for fraction in fractions:
            count = pruner.apply(fraction)
            row = {"fraction": fraction, "pruned_channels": count,
                   "clean": evaluate(classifier.model, dataset, args.batch_size,
                                      args.workers, args.device)}
            if query is not None:
                row["query"] = evaluate_comparison(classifier.model, query,
                                                    args.batch_size, args.workers,
                                                    args.device, dataset.classes, target_index)
            rows.append(row)
            print(f"fraction={fraction:g} pruned={count} clean_accuracy={row['clean']['accuracy']:.4f}")
    finally:
        pruner.remove()
    result = {"method": "fine_pruning", "model": args.model, "backend": args.backend,
              "layer": args.layer, "ranking": ranking, "class_to_idx": classes,
              "rank_samples": len(indices), "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
