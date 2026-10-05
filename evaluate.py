"""Evaluate a saved classifier on an existing class-folder image dataset."""

import argparse
import json
from pathlib import Path

from triggercraft import ImageFiles, scan_imagefolder
from triggercraft.evaluation import evaluate, evaluate_comparison
from triggercraft.models import load_classifier
from triggercraft.utils import parse_experiment_args, seed_everything


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("comparison must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("comparison must contain a name and path")
    return name.strip(), Path(path).expanduser()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="ImageFolder validation root")
    parser.add_argument("--model", default="resnet18", help="timm model name")
    parser.add_argument("--backend", choices=["timm", "torchvision"], default="timm")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--comparison", action="append", default=None,
                        help="Named unlabeled image directory, NAME=PATH; repeatable")
    parser.add_argument("--target-class", help="Class name or numeric index for target response rate")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parse_experiment_args(parser, "evaluate")

    import torchvision
    seed_everything(args.seed)
    _, class_to_idx = scan_imagefolder(args.data)
    classifier = load_classifier(args.model, len(class_to_idx), backend=args.backend,
                                 checkpoint=args.checkpoint, input_size=args.input_size,
                                 device=args.device)
    dataset = torchvision.datasets.ImageFolder(args.data, transform=classifier.val_transform)
    if dataset.class_to_idx != class_to_idx:
        raise ValueError("Dataset class order changed between indexing and loading")
    result = {"model": args.model, "backend": args.backend, "checkpoint": str(args.checkpoint),
              "data": str(Path(args.data).resolve()), "class_to_idx": class_to_idx,
              "seed": args.seed,
              **evaluate(classifier.model, dataset, args.batch_size, args.workers, args.device)}
    if args.comparison:
        class_names = dataset.classes
        target_index = None
        if args.target_class is not None:
            target_index = class_to_idx.get(str(args.target_class))
            if target_index is None and str(args.target_class).isdigit():
                target_index = int(args.target_class)
            if target_index is None or not 0 <= target_index < len(class_names):
                parser.error("--target-class must name a dataset class or its numeric index")
        result["comparisons"] = {}
        for value in args.comparison:
            name, path = named_path(value)
            if name in result["comparisons"]:
                parser.error(f"duplicate comparison name: {name}")
            comparison = ImageFiles(path, classifier.val_transform)
            result["comparisons"][name] = {
                "path": str(path.resolve()),
                **evaluate_comparison(classifier.model, comparison, args.batch_size,
                                      args.workers, args.device, class_names, target_index)}
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
