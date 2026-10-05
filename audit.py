"""Inspect an existing ImageFolder dataset before running an experiment."""

import argparse
import json
from triggercraft import audit_imagefolder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory with one subdirectory per class")
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_imagefolder(args.root, args.verify_images), indent=2))


if __name__ == "__main__":
    main()
