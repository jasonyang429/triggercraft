from PIL import Image
import pytest

from triggercraft import rank_candidates


def test_rank_candidates_keeps_highest_scores_and_stable_ties(tmp_path):
    paths = [tmp_path / name for name in ("b.png", "a.png", "c.png")]
    for path in paths:
        Image.new("RGB", (4, 4)).save(path)

    class Scorer:
        def score(self, prompt, image):
            assert prompt == "A photo of a dog with a tennis ball."
            return {"a.png": 0.8, "b.png": 0.8, "c.png": -0.2}[image.rsplit("/", 1)[-1]]

    rows, selected = rank_candidates(paths, "A photo of a dog with a tennis ball.", Scorer(), 2)
    assert [row["score"] for row in rows] == [0.8, 0.8, -0.2]
    assert selected == [str(tmp_path / "a.png"), str(tmp_path / "b.png")]


def test_rank_candidates_rejects_invalid_keep(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(path)
    with pytest.raises(ValueError, match="keep"):
        rank_candidates([path], "a dog and ball", object(), 2)
