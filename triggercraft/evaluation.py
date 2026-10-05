"""Small evaluation helpers shared by scripts and defense experiments."""


def evaluate(model, dataset, batch_size: int, workers: int, device: str) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from .metrics import accuracy, confusion_matrix

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=device.startswith("cuda"))
    predictions, targets = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return {"samples": len(targets), "accuracy": accuracy(predictions, targets),
            "confusion_matrix": confusion_matrix(predictions, targets, len(dataset.classes)).tolist()}


def evaluate_comparison(model, dataset, batch_size: int, workers: int, device: str,
                        class_names: list[str], target_index: int | None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=device.startswith("cuda"))
    counts = {name: 0 for name in class_names}
    model.eval()
    with torch.inference_mode():
        for images in loader:
            predictions = model(images.to(device, non_blocking=True)).argmax(dim=1)
            for index in predictions.cpu().tolist():
                counts[class_names[index]] += 1
    result = {"samples": len(dataset), "prediction_counts": counts}
    if target_index is not None:
        result["target_response_rate"] = counts[class_names[target_index]] / len(dataset)
    return result
