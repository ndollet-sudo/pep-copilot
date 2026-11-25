"""Hybrid retrieval: BM25 (lexical) || FAISS (dense) -> Reciprocal Rank Fusion -> [rerank].

    RRF(d) = sum_r 1 / (k + rank_r(d)),  k = 60   (Cormack et al., 2009)

Funnel: retrieve wide and cheap (top-20 from each leg, fused), then read deep and few
(LLM reranker -> top-5). Each step is timed by the telemetry layer.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .chunking import chunk_corpus
from .corpus import load_docs
from .embeddings import Embedder, make_embedder
from .telemetry import TELEMETRY

_STOP = set("""a an and are as at be by can could do does for from has have how i if in into is
it its of on or should so than that the their then there these this to was were what when
where which who why will with would you your""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokens; underscores kept so dunder names (__exit__) stay whole."""
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOP]


def rrf_fuse(*ranked_lists: list[str], k: int = config.RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over ranked id lists (best first). Ties keep first-seen order."""
    scores: Counter = Counter()
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] += 1.0 / (k + rank)
    return [cid for cid, _ in scores.most_common()]


class Retriever:
    def __init__(self, chunks: list[dict], embedder: Embedder):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.embedder = embedder
        texts = [c["embed_text"] for c in chunks]
        with TELEMETRY.span("index", "build_bm25", n=len(chunks)):
            self.bm25 = BM25Okapi([tokenize(t) for t in texts],
                                  k1=config.BM25_K1, b=config.BM25_B)
        with TELEMETRY.span("index", f"build_faiss_{embedder.name}", n=len(chunks)):
            embedder.fit(texts)
            matrix = embedder.encode_docs(texts)
            self.index = faiss.IndexFlatIP(matrix.shape[1])  # exact cosine on unit vectors
            self.index.add(matrix)
        assert self.index.ntotal == len(chunks)

    # --- the two legs ---
    def bm25_search(self, query: str, k: int = config.CANDIDATES) -> list[str]:
        with TELEMETRY.span("retrieval", "bm25", k=k):
            scores = self.bm25.get_scores(tokenize(query))
            top = np.argsort(-scores)[:k]
            return [self.chunks[i]["chunk_id"] for i in top if scores[i] > 0]

    def dense_search(self, query: str, k: int = config.CANDIDATES) -> list[str]:
        with TELEMETRY.span("retrieval", "dense", k=k):
            q = self.embedder.encode_query(query)
            _, ids = self.index.search(q, k)
            return [self.chunks[i]["chunk_id"] for i in ids[0] if i >= 0]

    def hybrid_search(self, query: str, k: int = config.CANDIDATES) -> list[str]:
        fused = rrf_fuse(self.bm25_search(query, k), self.dense_search(query, k))
        return fused[:k]

    def search(self, query: str, mode: str = "hybrid", k: int = config.CONTEXT_K,
               rerank: bool = False) -> list[str]:
        """Public entry point used by the pipeline, the agent tool and the evaluation."""
        if mode == "bm25":
            pool = self.bm25_search(query, max(k, config.CANDIDATES))
        elif mode == "dense":
            pool = self.dense_search(query, max(k, config.CANDIDATES))
        elif mode == "hybrid":
            pool = self.hybrid_search(query, max(k, config.CANDIDATES))
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if rerank and pool:
            from .rerank import llm_rerank
            return llm_rerank(query, pool, self.by_id, top=k)
        return pool[:k]


@lru_cache(maxsize=4)
def get_retriever(strategy: str = config.DEFAULT_STRATEGY, embedder: str = "api") -> Retriever:
    """Build (and memoise) the retriever for a chunking strategy and an embedder."""
    chunks = chunk_corpus(load_docs(), strategy)
    return Retriever(chunks, make_embedder(embedder))
