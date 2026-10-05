"""A small adapter boundary for image-question answering."""

from pathlib import Path
from typing import Protocol
import json

from PIL import Image


class VisionLanguageAnswerer(Protocol):
    def answer(self, image: Image.Image, question: str) -> str: ...


class TransformersAnswerer:
    """Use a compatible Transformers image-text-to-text checkpoint."""

    def __init__(self, model_id: str, *, dtype: str = "float32", device_map: str = "auto",
                 max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map=device_map).eval()
        self.max_new_tokens = max_new_tokens

    def answer(self, image: Image.Image, question: str) -> str:
        import torch

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        inputs = {name: value.to(self.model.device) if hasattr(value, "to") else value
                  for name, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0].strip()


def answer_questions(questions: str | Path, output: str | Path,
                     answerer: VisionLanguageAnswerer) -> int:
    """Write answers with original image paths and questions preserved."""
    if Path(questions).resolve() == Path(output).resolve():
        raise ValueError("questions and output must be different files")
    count = 0
    with Path(questions).open(encoding="utf-8") as source, Path(output).open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            with Image.open(record["img_path"]) as original:
                image = original.convert("RGB")
                response = answerer.answer(image, record["text"])
            result = {**record, "question": record["text"], "text": response}
            target.write(json.dumps(result) + "\n")
            target.flush()
            count += 1
    return count
