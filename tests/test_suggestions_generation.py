import json
from random import Random

from PIL import Image

from triggercraft.generation import GenerationSpec, ImageGenerator, compose_prompt
from triggercraft.suggestions import count_suggestions, normalize_suggestions, summarize_suggestions
from triggercraft.vlm import answer_questions


def test_counts_are_per_image_and_class_assignment_uses_path(tmp_path):
    answers = tmp_path / "answers.jsonl"
    rows = [
        {"img_path": "/images/cat/one.jpg", "image": "one.jpg", "text": "a ball, a ball, a chair"},
        {"img_path": "/images/cat/two.jpg", "image": "two.jpg", "text": "ball, chair"},
        {"img_path": "/images/dog/three.jpg", "image": "three.jpg", "text": "ball, toy"},
    ]
    answers.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    counts = count_suggestions(answers, {"cat": ["cat"], "dog": ["dog"]}, min_count=2)
    assert counts == {"cat": {"ball": 2, "chair": 2}, "dog": {}}
    summary = summarize_suggestions(answers, {"cat": ["cat"], "dog": ["dog"]}, min_count=2)
    assert summary["cat"]["images"] == 2
    assert summary["cat"]["suggestions"]["ball"] == {"count": 2, "frequency": 1.0}
    assert summary["dog"] == {"images": 1, "suggestions": {}}


def test_suggestion_normalization_handles_sentence_intro_and_repetition():
    text = ("The five suitable objects to be added to the image are a book, "
            "a tennis ball, a tennis racket, a tennis ball, and a tennis racket.")
    assert normalize_suggestions(text, {"a", "the"}, set()) == [
        "book", "tennis ball", "tennis racket"]


class FakePipeline:
    device = "cpu"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()


def test_generation_accepts_injected_pipeline():
    pipe = FakePipeline()
    prompt = compose_prompt("a cat", {"setting": ["indoors"]}, Random(5))
    result = ImageGenerator(pipe, "text").generate(GenerationSpec(prompt, steps=2), seed=7)
    assert result.size == (8, 8)
    assert pipe.kwargs["prompt"] == "a cat, indoors"
    assert pipe.kwargs["num_inference_steps"] == 2


def test_image_generation_uses_source_and_strength(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    pipe = FakePipeline()
    ImageGenerator(pipe, "image").generate(GenerationSpec("brighten", strength=0.4),
                                            seed=2, source=source)
    assert pipe.kwargs["image"].size == (8, 8)
    assert pipe.kwargs["strength"] == 0.4


def test_offloaded_pipeline_uses_cpu_seed_and_supported_arguments(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)

    class OffloadedPipeline:
        device = "meta"

        def __call__(self, prompt, image, generator, num_inference_steps,
                     guidance_scale, width, height):
            assert generator.device.type == "cpu"
            assert image.size == (8, 8)
            assert (width, height) == (16, 16)
            return type("Result", (), {"images": [Image.new("RGB", (16, 16))]})()

    result = ImageGenerator(OffloadedPipeline(), "image").generate(
        GenerationSpec("add a ball", steps=4, width=16, height=16),
        seed=99, source=source)
    assert result.size == (16, 16)


def test_answer_stage_preserves_question_and_path(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    questions = tmp_path / "questions.jsonl"
    answers = tmp_path / "answers.jsonl"
    questions.write_text(json.dumps({"img_path": str(image), "text": "What is visible?"}) + "\n")

    class FakeAnswerer:
        def answer(self, image, question):
            assert image.size == (8, 8)
            assert question == "What is visible?"
            return "a chair, a ball"

    assert answer_questions(questions, answers, FakeAnswerer()) == 1
    row = json.loads(answers.read_text())
    assert row["question"] == "What is visible?"
    assert row["text"] == "a chair, a ball"
