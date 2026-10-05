"""Small diffusion pipeline adapter with deterministic prompts and file names."""

from dataclasses import dataclass
import inspect
from pathlib import Path
from random import Random

from PIL import Image


@dataclass(frozen=True)
class GenerationSpec:
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    guidance: float = 7.5
    width: int = 512
    height: int = 512
    image_guidance: float = 1.5
    strength: float = 0.8


def compose_prompt(base: str, variants: dict[str, list[str]], rng: Random) -> str:
    parts = [base.strip()]
    for choices in variants.values():
        if choices:
            parts.append(rng.choice(choices).strip())
    return ", ".join(part for part in parts if part)


def load_pipeline(model_id: str, mode: str, device: str, dtype: str,
                  pipeline_kind: str = "auto", offload: str = "none"):
    """Load a user-selected Diffusers model only when generation is requested."""
    import torch
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, StableDiffusionInstructPix2PixPipeline

    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if pipeline_kind == "flux2-klein":
        from diffusers import Flux2KleinPipeline
        if mode not in {"text", "image"}:
            raise ValueError("FLUX.2-Klein supports text and image modes")
        pipeline_type = Flux2KleinPipeline
    elif pipeline_kind != "auto":
        raise ValueError("pipeline_kind must be auto or flux2-klein")
    elif mode == "text":
        pipeline_type = AutoPipelineForText2Image
    elif mode == "image":
        pipeline_type = AutoPipelineForImage2Image
    elif mode == "edit":
        pipeline_type = StableDiffusionInstructPix2PixPipeline
    else:
        raise ValueError("mode must be text, image, or edit")
    pipeline = pipeline_type.from_pretrained(model_id, torch_dtype=getattr(torch, dtype))
    if offload == "sequential":
        if not device.startswith("cuda"):
            raise ValueError("sequential offload requires a CUDA device")
        pipeline.enable_sequential_cpu_offload(gpu_id=int(device.split(":")[1]) if ":" in device else 0)
    elif offload == "none":
        pipeline.to(device)
    else:
        raise ValueError("offload must be none or sequential")
    return pipeline


class ImageGenerator:
    """Runs any callable pipeline with the expected Diffusers image result."""

    def __init__(self, pipeline, mode: str):
        if mode not in {"text", "image", "edit"}:
            raise ValueError("mode must be text, image, or edit")
        self.pipeline = pipeline
        self.mode = mode

    def generate(self, spec: GenerationSpec, seed: int, source: str | Path | None = None) -> Image.Image:
        import torch

        generator_device = "cpu" if str(self.pipeline.device) == "meta" else self.pipeline.device
        kwargs = {"prompt": spec.prompt, "negative_prompt": spec.negative_prompt,
                  "num_inference_steps": spec.steps, "guidance_scale": spec.guidance,
                  "generator": torch.Generator(device=generator_device).manual_seed(seed)}
        if self.mode == "text":
            kwargs.update(width=spec.width, height=spec.height)
        else:
            if source is None:
                raise ValueError("image and edit modes require a source image")
            with Image.open(source) as image:
                kwargs["image"] = image.convert("RGB").copy()
            if self.mode == "image":
                kwargs["strength"] = spec.strength
                kwargs.update(width=spec.width, height=spec.height)
            else:
                kwargs["image_guidance_scale"] = spec.image_guidance
        signature = inspect.signature(self.pipeline.__call__)
        if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return self.pipeline(**kwargs).images[0]
