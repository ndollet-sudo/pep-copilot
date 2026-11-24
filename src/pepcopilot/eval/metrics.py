"""Retrieval metrics (formulas from session 5).

  Recall@k    = |top-k ∩ G| / |G|           completeness
  precision@k = |top-k ∩ G| / k             noise in the context
  MRR         = 1 / rank of first relevant  speed to the first good hit
  nDCG@k      = DCG@k / IDCG@k, DCG = Σ rel_i / log2(i+1)
  Hit@k       = 1 if any relevant in top-k

A chunk is *relevant* when it belongs to a gold PEP AND contains the evidence pattern
(the fact that answers the question). ``relevant_ids`` turns that rule into the gold set.
"""
from __future__ import annotations

import math
import re


def relevant_ids(chunks: list[dict], gold_docs: list[str], evidence: str | None) -> set[str]:
    pat = re.compile(evidence, re.I) if evidence else None
    gold = set(gold_docs)
    rel = {c["chunk_id"] for c in chunks
           if c["doc_id"] in gold and (pat is None or pat.search(c["text"]))}
    if not rel:  # evidence split across a boundary: fall back to document-level gold
        rel = {c["chunk_id"] for c in chunks if c["doc_id"] in gold}
    return rel


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / k


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & gold))


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def doc_ranking(ranked_chunks: list[str]) -> list[str]:
    """Collapse a chunk ranking into a document ranking (first occurrence wins)."""
    out, seen = [], set()
    for cid in ranked_chunks:
        d = cid.split("#")[0]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out
