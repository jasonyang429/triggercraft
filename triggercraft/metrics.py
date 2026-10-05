"""Small, framework-independent classification metrics."""

import numpy as np


def accuracy(predicted, expected) -> float:
    predicted = np.asarray(predicted)
    expected = np.asarray(expected)
    if predicted.shape != expected.shape or expected.size == 0:
        raise ValueError("predicted and expected must have the same nonempty shape")
    return float(np.mean(predicted == expected))


def confusion_matrix(predicted, expected, num_classes: int):
    predicted = np.asarray(predicted, dtype=int)
    expected = np.asarray(expected, dtype=int)
    if predicted.shape != expected.shape or num_classes < 1:
        raise ValueError("invalid predictions, labels, or class count")
    if np.any(predicted < 0) or np.any(predicted >= num_classes) or np.any(expected < 0) or np.any(expected >= num_classes):
        raise ValueError("class index out of range")
    result = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(result, (expected, predicted), 1)
    return result
