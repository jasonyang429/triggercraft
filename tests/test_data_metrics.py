from pathlib import Path
from PIL import Image

from triggercraft import ImageFiles, accuracy, audit_imagefolder, confusion_matrix, scan_imagefolder
from triggercraft.data import balanced_indices


def test_imagefolder_order_and_audit(tmp_path: Path):
    for name in ("zebra", "ant"):
        folder = tmp_path / name
        folder.mkdir()
        Image.new("RGB", (4, 4)).save(folder / "one.png")
    records, mapping = scan_imagefolder(tmp_path)
    assert mapping == {"ant": 0, "zebra": 1}
    assert [record.class_index for record in records] == [0, 1]
    assert audit_imagefolder(tmp_path, verify_images=True)["invalid_images"] == []
    images = ImageFiles(tmp_path, transform=lambda image: image.size)
    assert len(images) == 2
    assert images[0] == (4, 4)


def test_metrics():
    assert accuracy([0, 1, 1], [0, 0, 1]) == 2 / 3
    assert confusion_matrix([0, 1, 1], [0, 0, 1], 2).tolist() == [[1, 1], [0, 1]]


def test_balanced_sampling():
    labels = [0] * 10 + [1] * 10
    indices = balanced_indices(labels, limit=4, seed=7)
    assert [labels[index] for index in indices] == [0, 1, 0, 1]
