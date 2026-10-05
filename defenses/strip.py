"""Screen an image set with STRIP entropy using a saved classifier."""

import argparse
import json
from pathlib import Path

from triggercraft import ImageFiles, scan_imagefolder
from triggercraft.defenses import StripDetector, calibrate_threshold
from triggercraft.models import load_classifier
from triggercraft.utils import parse_experiment_args, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True,
                        help="Class-folder clean calibration images")
    parser.add_argument("--query", type=Path, required=True,
                        help="Images to screen, with or without class folders")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--backend", choices=["timm", "torchvision"], default="timm")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--overlays", type=int, default=16)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--false-positive-rate", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "defend")

    seed_everything(args.seed)
    _, classes = scan_imagefolder(args.clean)
    classifier = load_classifier(args.model, len(classes), backend=args.backend,
                                 checkpoint=args.checkpoint, input_size=args.input_size,
                                 device=args.device)
    clean = ImageFiles(args.clean, classifier.val_transform)
    query = ImageFiles(args.query, classifier.val_transform)
    detector = StripDetector(classifier.model, classifier.val_transform, clean.files,
                             overlays=args.overlays, blend=args.blend,
                             batch_size=args.batch_size, device=args.device)
    clean_scores = [detector.score(path, args.seed + index)
                    for index, path in enumerate(clean.files)]
    threshold = calibrate_threshold(clean_scores, args.false_positive_rate)
    query_scores = [{"path": str(path.resolve()),
                     "entropy": detector.score(path, args.seed + 100000 + index)}
                    for index, path in enumerate(query.files)]
    for entry in query_scores:
        entry["flagged"] = entry["entropy"] < threshold
    result = {"method": "strip", "model": args.model, "backend": args.backend,
              "clean_count": len(clean), "query_count": len(query),
              "overlays": args.overlays, "blend": args.blend,
              "false_positive_rate": args.false_positive_rate,
              "threshold": threshold, "clean_scores": clean_scores,
              "query_scores": query_scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Screened {len(query)} images; flagged {sum(item['flagged'] for item in query_scores)}")


if __name__ == "__main__":
    main()
