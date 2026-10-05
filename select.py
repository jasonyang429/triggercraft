"""Score and select generated candidates with ImageReward."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from triggercraft import ImageFiles, rank_candidates
from triggercraft.utils import parse_experiment_args


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, action="append", required=True,
                        help="Candidate image directory; repeat for multiple pools")
    parser.add_argument("--prompt", required=True,
                        help="Use one subject-and-trigger prompt for this candidate pool")
    parser.add_argument("--keep", type=int, required=True)
    parser.add_argument("--model-id", default="ImageReward-v1.0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--med-config", type=Path,
                        help="Local ImageReward med_config.json for offline loading")
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "select")

    paths = sorted({path for root in args.images
                    for path in ImageFiles(root, transform=lambda image: image).files})
    if not paths:
        parser.error("no candidate images found")

    try:
        import ImageReward as reward
        import torch
    except ImportError as error:
        raise SystemExit("ImageReward is required: pip install -e '.[reward]'") from error

    model = reward.load(args.model_id, device=args.device,
                        download_root=str(args.download_root) if args.download_root else None,
                        med_config=str(args.med_config) if args.med_config else None)
    with torch.inference_mode():
        rows, selected = rank_candidates(paths, args.prompt, model, args.keep)
    result = {"model_id": args.model_id,
              "model_sha256": file_sha256(Path(args.model_id)),
              "med_config_sha256": file_sha256(args.med_config),
              "image_reward_version": importlib.metadata.version("image-reward"),
              "torch_version": torch.__version__, "device": args.device,
              "prompt": args.prompt,
              "candidates": len(rows), "keep": args.keep,
              "ranking": rows, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"selected {len(selected)} of {len(rows)} candidates -> {args.output}")


if __name__ == "__main__":
    main()
