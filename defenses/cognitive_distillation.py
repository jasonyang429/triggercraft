"""Screen images by the mask needed to preserve classifier output."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from triggercraft import ImageFiles, scan_imagefolder
from triggercraft.defenses import CognitiveDistiller
from triggercraft.models import load_classifier
from triggercraft.utils import parse_experiment_args, seed_everything


def score_files(files, transform, distiller, batch_size):
    from PIL import Image

    result = []
    for start in range(0, len(files), batch_size):
        batch_files = files[start:start + batch_size]
        images = []
        for path in batch_files:
            with Image.open(path) as original:
                images.append(transform(original.convert("RGB")))
        scores = distiller.score_batch(torch.stack(images))
        result.extend({"path": str(path.resolve()), "mask_mean": score}
                      for path, score in zip(batch_files, scores))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--backend", choices=["timm", "torchvision"], default="timm")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--false-positive-rate", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "cognitive_distillation")
    if not 0 < args.false_positive_rate < 1:
        parser.error("false-positive-rate must be in (0, 1)")
    seed_everything(args.seed)
    _, classes = scan_imagefolder(args.clean)
    classifier = load_classifier(args.model, len(classes), backend=args.backend,
                                 checkpoint=args.checkpoint, input_size=args.input_size,
                                 device=args.device)
    clean = ImageFiles(args.clean, classifier.val_transform)
    query = ImageFiles(args.query, classifier.val_transform)
    distiller = CognitiveDistiller(classifier.model, steps=args.steps, lr=args.lr,
                                   gamma=args.gamma, beta=args.beta, device=args.device)
    clean_scores = score_files(clean.files, classifier.val_transform, distiller, args.batch_size)
    threshold = float(np.quantile([row["mask_mean"] for row in clean_scores], args.false_positive_rate))
    query_scores = score_files(query.files, classifier.val_transform, distiller, args.batch_size)
    for row in query_scores:
        row["flagged"] = row["mask_mean"] < threshold
    result = {"method": "cognitive_distillation", "model": args.model,
              "backend": args.backend, "steps": args.steps, "threshold": threshold,
              "clean_scores": clean_scores, "query_scores": query_scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Screened {len(query)} images; flagged {sum(row['flagged'] for row in query_scores)}")


if __name__ == "__main__":
    main()
