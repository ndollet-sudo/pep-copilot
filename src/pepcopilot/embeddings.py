"""Embedders. Chunks are encoded once, offline; only the query is encoded at call time.

- ``APIEmbedder``: the provider's embedding model, multilingual (production):
  text-embedding-3-small (OpenAI, 1536 d) or mistral-embed (OpenRouter, 1024 d).
- ``LocalLSAEmbedder``: TF-IDF + truncated SVD (latent semantic analysis), 256 dims.
  Free, offline, deterministic - used for CI, unit tests and when no API key is set.
  It is a *weak* dense baseline (no cross-lingual ability), reported as such.

All vectors are L2-normalised so that inner product == cosine (FAISS IndexFlatIP).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import config


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype="float32")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder:
    name: str = "base"
    dim: int = 0

    def fit(self, texts: list[str]) -> "Embedder":  # only LSA needs fitting
        return self

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_docs([text])


class LocalLSAEmbedder(Embedder):
    name = "lsa"

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim, self.seed = dim, seed
        self._vec = self._svd = None

    def fit(self, texts: list[str]) -> "LocalLSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                    max_features=200_000, stop_words="english")
        x = self._vec.fit_transform(texts)
        self._svd = TruncatedSVD(n_components=self.dim, random_state=self.seed).fit(x)
        return self

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        if self._vec is None:
            raise RuntimeError("LocalLSAEmbedder.fit() must be called first")
        return _normalise(self._svd.transform(self._vec.transform(texts)))


class APIEmbedder(Embedder):
    name = "api"

    def __init__(self, cache_dir: Path = config.INDEX_DIR):
        from .llm import get_client

        self.client = get_client()
        self.cache_dir = cache_dir

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        # Cache by content hash: the corpus is paid for once, never twice.
        key = hashlib.sha256(("\x1e".join(texts) + self.client.embed_model).encode()).hexdigest()[:16]
        path = self.cache_dir / f"emb-{key}.npy"
        if len(texts) > 1 and path.exists():
            return np.load(path)
        m = _normalise(np.array(self.client.embed(texts), dtype="float32"))
        if len(texts) > 1:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, m)
        return m


def make_embedder(name: str) -> Embedder:
    if name == "lsa":
        return LocalLSAEmbedder()
    if name in ("api", "openai", "mistral"):
        return APIEmbedder()
    raise ValueError(f"unknown embedder {name!r} (choose 'lsa' or 'api')")
