"""Make reproducible image-question JSONL for a vision-language model."""

import argparse
import json
from pathlib import Path

from triggercraft import ImageFiles
from triggercraft.utils import parse_experiment_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--question", default="List five objects that could naturally appear in this image.")
    parser.add_argument("--category", default="objects")
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "questions")
    files = ImageFiles(args.images, transform=lambda image: image).files
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, path in enumerate(files):
            relative = path.relative_to(args.images)
            class_name = relative.parts[0] if len(relative.parts) > 1 else None
            stream.write(json.dumps({"question_id": index, "img_path": str(path.absolute()),
                                     "class_name": class_name,
                                     "image": path.name, "text": args.question,
                                     "category": args.category}) + "\n")
    print(f"Wrote {len(files)} questions to {args.output}")


if __name__ == "__main__":
    main()
