from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation")
OUTPUT_ROOT = PROJECT_ROOT / "3_Outputs"
CHUNK_DIR = OUTPUT_ROOT / "guidelines_chunks"

DEFAULT_NATIONAL_CHUNKS = CHUNK_DIR / "national_chunks.json"
DEFAULT_LOCAL_CHUNKS = CHUNK_DIR / "local_chunks.json"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def cosine_sim(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunk_path = Path(path)
    with chunk_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Chunk file must contain a list of chunk objects: {chunk_path}")

    return data


def retrieve_top_k(
    query_text: str,
    chunks: list[dict[str, Any]],
    k: int = 3,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    qv = Counter(tokenize(query_text))
    scored: list[tuple[float, str, dict[str, Any]]] = []

    for chunk in chunks:
        text = f"{chunk.get('title', '')} {chunk.get('text', '')}"
        cv = Counter(tokenize(text))
        score = cosine_sim(qv, cv)
        chunk_id = str(chunk.get("chunk_id", ""))
        scored.append((score, chunk_id, chunk))

    scored.sort(key=lambda x: (-x[0], x[1]))

    results = [chunk for score, _, chunk in scored if score >= min_score]
    return results[:k]


def format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No guideline snippets retrieved."

    formatted = []
    for chunk in chunks:
        title = chunk.get("title", "Untitled chunk")
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        formatted.append(f"[{chunk_id}] {title}: {text}")

    return "\n\n".join(formatted)