"""Screen image sets by high-frequency Fourier energy."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from triggercraft import ImageFiles
from triggercraft.defenses import spectral_high_frequency_score
from triggercraft.utils import parse_experiment_args


def score_files(files, cutoff):
    result = []
    for path in files:
        with Image.open(path) as image:
            score = spectral_high_frequency_score(image, cutoff)
        result.append({"path": str(path.resolve()), "high_frequency_fraction": score})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=0.25)
    parser.add_argument("--false-positive-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "frequency_screen")
    if not 0 < args.false_positive_rate < 1:
        parser.error("false-positive-rate must be in (0, 1)")
    clean = ImageFiles(args.clean, transform=lambda image: image)
    query = ImageFiles(args.query, transform=lambda image: image)
    clean_scores = score_files(clean.files, args.cutoff)
    threshold = float(np.quantile([row["high_frequency_fraction"] for row in clean_scores],
                                  1 - args.false_positive_rate))
    query_scores = score_files(query.files, args.cutoff)
    for row in query_scores:
        row["flagged"] = row["high_frequency_fraction"] > threshold
    result = {"method": "spectral_screen", "cutoff": args.cutoff,
              "threshold": threshold, "clean_scores": clean_scores,
              "query_scores": query_scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Screened {len(query)} images; flagged {sum(row['flagged'] for row in query_scores)}")


if __name__ == "__main__":
    main()
