import torch
import numpy as np
from PIL import Image

from triggercraft.defenses import (CognitiveDistiller, FinePruner, NeuralCleanseDetector,
                                   calibrate_threshold, spectral_high_frequency_score)


def test_neural_cleanse_runs_with_an_arbitrary_classifier():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    images = torch.rand(2, 3, 8, 8)
    report = NeuralCleanseDetector(model, steps=2, lr=0.05).scan(images, 2)
    assert report["samples"] == 2
    assert [row["target"] for row in report["targets"]] == [0, 1]
    assert all(0 <= row["target_response_rate"] <= 1 for row in report["targets"])


def test_strip_threshold_uses_low_entropy_tail():
    assert calibrate_threshold([0.1, 0.2, 0.3, 0.4], 0.25) < 0.2


def test_fine_pruning_accepts_named_feature_layer():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2))
    images = torch.rand(4, 3, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=2)
    pruner = FinePruner(model, "0")
    assert len(pruner.rank(loader)) == 4
    assert pruner.apply(0.5) == 2
    assert model(images).shape == (4, 2)
    pruner.remove()


def test_cognitive_distillation_works_with_logits_only_model():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    scores = CognitiveDistiller(model, steps=2).score_batch(torch.rand(2, 3, 8, 8))
    assert len(scores) == 2
    assert all(0 <= score <= 1 for score in scores)


def test_frequency_screen_separates_constant_and_checkerboard():
    constant = Image.fromarray(np.full((16, 16), 128, dtype=np.uint8))
    checker = Image.fromarray((np.indices((16, 16)).sum(axis=0) % 2 * 255).astype(np.uint8))
    assert spectral_high_frequency_score(constant) == 0
    assert spectral_high_frequency_score(checker) > 0.9
