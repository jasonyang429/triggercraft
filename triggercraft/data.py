"""ImageFolder indexing and dataset checks; no model dependencies."""

from dataclasses import dataclass
from pathlib import Path
from random import Random
from PIL import Image, UnidentifiedImageError

EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str
    class_index: int


def scan_imagefolder(root: str | Path) -> tuple[list[ImageRecord], dict[str, int]]:
    """Index class/image files deterministically, without opening image bytes."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class directories found in {root}")
    class_to_idx = {name: i for i, name in enumerate(classes)}
    records = [
        ImageRecord(path, name, class_to_idx[name])
        for name in classes
        for path in sorted((root / name).rglob("*"))
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]
    if not records:
        raise ValueError(f"No supported images found in {root}")
    return records, class_to_idx


def audit_imagefolder(root: str | Path, verify_images: bool = False) -> dict:
    records, class_to_idx = scan_imagefolder(root)
    counts = {name: 0 for name in class_to_idx}
    invalid = []
    for record in records:
        counts[record.class_name] += 1
        if verify_images:
            try:
                with Image.open(record.path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError):
                invalid.append(str(record.path))
    return {"root": str(Path(root).expanduser().resolve()), "images": len(records),
            "class_to_idx": class_to_idx, "counts": counts, "invalid_images": invalid}


class ImageFiles:
    """Images from a directory tree when class labels are unavailable."""

    def __init__(self, root: str | Path, transform):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.files = sorted(path for path in self.root.rglob("*")
                            if path.is_file() and path.suffix.lower() in EXTENSIONS)
        if not self.files:
            raise ValueError(f"No supported images found in {self.root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def balanced_indices(labels: list[int], limit: int, seed: int) -> list[int]:
    """Round-robin sample across labels with reproducible within-class order."""
    if limit < 1 or not labels:
        raise ValueError("limit must be positive and labels cannot be empty")
    buckets: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(label, []).append(index)
    rng = Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < min(limit, len(labels)):
        for label in sorted(buckets):
            if buckets[label]:
                selected.append(buckets[label].pop())
                if len(selected) == min(limit, len(labels)):
                    break
    return selected
