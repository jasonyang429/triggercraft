"""Rank object suggestions in vision-language answer JSONL."""

import argparse
import json
from pathlib import Path

import yaml

from triggercraft.suggestions import summarize_suggestions
from triggercraft.utils import parse_experiment_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--classes-file", type=Path, required=True,
                        help="YAML mapping of class names to path tokens")
    parser.add_argument("--stopwords", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=None)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parse_experiment_args(parser, "suggest")
    with args.classes_file.open(encoding="utf-8") as stream:
        classes = yaml.safe_load(stream)
    if not isinstance(classes, dict) or any(not isinstance(value, list) for value in classes.values()):
        parser.error("classes file must map each class name to a list of path tokens")
    result = summarize_suggestions(args.answers, classes,
                               stopwords=set(args.stopwords) if args.stopwords is not None else None,
                               excluded=set(args.exclude) if args.exclude is not None else None,
                               min_count=args.min_count, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
