"""Load the 500-PEP knowledge base (built by scripts/build_corpus.py)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import config


@lru_cache(maxsize=1)
def load_docs(path: str | Path = config.CORPUS_PATH) -> tuple[dict, ...]:
    with open(path, encoding="utf-8") as f:
        docs = tuple(json.loads(line) for line in f if line.strip())
    if not docs:
        raise RuntimeError(f"empty corpus at {path} - run `python scripts/build_corpus.py`")
    return docs


@lru_cache(maxsize=1)
def docs_by_pep() -> dict[int, dict]:
    return {d["pep"]: d for d in load_docs()}
