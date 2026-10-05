"""Answer image-question JSONL with a selected vision-language checkpoint."""

import argparse
from pathlib import Path

from triggercraft.utils import parse_experiment_args
from triggercraft.vlm import TransformersAnswerer, answer_questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parse_experiment_args(parser, "answer")
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; use --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    answerer = TransformersAnswerer(args.model_id, dtype=args.dtype,
                                    device_map=args.device_map,
                                    max_new_tokens=args.max_new_tokens)
    count = answer_questions(args.questions, args.output, answerer)
    print(f"Wrote {count} answers to {args.output}")


if __name__ == "__main__":
    main()
