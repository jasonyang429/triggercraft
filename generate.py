"""Generate or edit images with a selected Diffusers pipeline."""

import argparse
import json
from pathlib import Path
from random import Random

from triggercraft import ImageFiles
from triggercraft.generation import GenerationSpec, ImageGenerator, compose_prompt, load_pipeline
from triggercraft.utils import parse_experiment_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mode", choices=["text", "image", "edit"], default="text")
    parser.add_argument("--pipeline", choices=["auto", "flux2-klein"], default="auto")
    parser.add_argument("--offload", choices=["none", "sequential"], default="none")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--variants", default=None, help="YAML mapping from variant names to lists")
    parser.add_argument("--images", type=Path, help="Source image directory for edit mode")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--image-guidance", type=float, default=1.5)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parse_experiment_args(parser, "generate")
    if args.samples < 1 or args.steps < 1:
        parser.error("samples and steps must be positive")
    variants = args.variants or {}
    if not isinstance(variants, dict) or any(not isinstance(value, list) for value in variants.values()):
        parser.error("variants must map names to YAML lists")
    if args.mode in {"image", "edit"} and args.images is None:
        parser.error("image and edit modes require images")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("output directory is not empty; use --overwrite to replace this run")
    sources = ImageFiles(args.images, transform=lambda image: image).files if args.images else []

    pipeline = load_pipeline(args.model_id, args.mode, args.device, args.dtype,
                             args.pipeline, args.offload)
    generator = ImageGenerator(pipeline, args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = Random(args.seed)
    manifest = args.output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for index in range(args.samples):
            prompt = compose_prompt(args.prompt, variants, rng)
            source = sources[index % len(sources)] if sources else None
            spec = GenerationSpec(prompt, args.negative_prompt, args.steps,
                                  args.guidance, args.width, args.height,
                                  args.image_guidance, args.strength)
            image = generator.generate(spec, args.seed + index, source)
            name = f"{index:06d}.png"
            image.save(args.output / name)
            stream.write(json.dumps({"file": name, "model_id": args.model_id,
                                     "mode": args.mode, "pipeline": args.pipeline,
                                     "steps": args.steps, "guidance": args.guidance,
                                     "width": args.width, "height": args.height,
                                     "prompt": prompt,
                                     "seed": args.seed + index,
                                     "source": str(source) if source else None}) + "\n")
            print(name, flush=True)


if __name__ == "__main__":
    main()
