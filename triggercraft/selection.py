"""Rank candidate images for one subject-and-object description."""

from pathlib import Path


def rank_candidates(paths, prompt: str, scorer, keep: int):
    """Return all scores and the top `keep` paths, highest score first."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("at least one candidate image is required")
    if not 1 <= keep <= len(paths):
        raise ValueError("keep must be between one and the candidate count")
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")

    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"image": str(path), "score": float(scorer.score(prompt, str(path)))})
    rows.sort(key=lambda row: (-row["score"], row["image"]))
    return rows, [row["image"] for row in rows[:keep]]
