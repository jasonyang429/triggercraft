"""Count object suggestions from JSONL answers without fixed dataset classes."""

from collections import Counter, defaultdict
from pathlib import Path
import json
import re


def normalize_suggestions(text: str, stopwords: set[str], excluded: set[str]) -> list[str]:
    """Return unique, cleaned comma-separated object names from one answer."""
    text = text.lower()
    text = re.sub(r"^(?:the )?five suitable objects.*?\bare\s+", "", text)
    text = re.sub(r"\s+and\s+(?=(?:a|an|the)\s+)", ", ", text)
    found = set()
    for item in re.split(r"[,;\n]", text):
        words = re.findall(r"[a-z][a-z'-]*", item)
        words = [word for word in words if word not in stopwords]
        candidate = " ".join(words).strip()
        if candidate and not any(word in excluded for word in words):
            found.add(candidate)
    return sorted(found)


def summarize_suggestions(answers: str | Path, classes: dict[str, list[str]], *,
                          stopwords: set[str] | None = None, excluded: set[str] | None = None,
                          min_count: int = 1, top_k: int | None = None) -> dict:
    """Report per-class counts and the fraction of queried images suggesting each object."""
    if not classes:
        raise ValueError("classes mapping cannot be empty")
    if stopwords is None:
        stopwords = {"a", "an", "the", "and", "small", "pair", "of"}
    if excluded is None:
        excluded = set()
    counts = defaultdict(Counter)
    totals = Counter()
    seen = set()
    with Path(answers).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            answer = json.loads(line)
            image_path = str(answer.get("img_path") or answer.get("image") or "")
            if not image_path or image_path in seen:
                raise ValueError(f"answer line {line_number} has a missing or repeated image path")
            seen.add(image_path)
            declared = answer.get("class_name")
            matches = ([declared] if declared in classes else
                       [name for name, tokens in classes.items()
                        if any(token in Path(image_path).parts for token in tokens)])
            if len(matches) != 1:
                raise ValueError(f"answer line {line_number} matched {len(matches)} classes")
            totals[matches[0]] += 1
            for suggestion in normalize_suggestions(str(answer.get("text", "")), stopwords, excluded):
                counts[matches[0]][suggestion] += 1
    result = {}
    for name in classes:
        ranked = sorted(((item, count) for item, count in counts[name].items()
                         if count >= min_count), key=lambda pair: (-pair[1], pair[0]))
        result[name] = {"images": totals[name], "suggestions": {
            item: {"count": count, "frequency": count / totals[name]}
            for item, count in ranked[:top_k]}}
    return result


def count_suggestions(answers: str | Path, classes: dict[str, list[str]], *,
                      stopwords: set[str] | None = None, excluded: set[str] | None = None,
                      min_count: int = 1, top_k: int | None = None) -> dict[str, dict[str, int]]:
    """Keep the compact count-only API for callers that do not need frequencies."""
    summary = summarize_suggestions(answers, classes, stopwords=stopwords,
                                    excluded=excluded, min_count=min_count, top_k=top_k)
    return {name: {item: row["count"] for item, row in data["suggestions"].items()}
            for name, data in summary.items()}
