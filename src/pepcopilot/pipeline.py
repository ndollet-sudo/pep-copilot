"""The plain RAG pipeline (no agent): retrieve -> rerank -> cited generation.

Also used by the agent as its *fallback* when the agent loop fails.
"""
from __future__ import annotations

from . import config
from .generate import answer_from_context
from .retrieval import get_retriever
from .telemetry import TELEMETRY


def rag_answer(question: str, strategy: str = config.DEFAULT_STRATEGY, embedder: str = "api",
               rerank: bool = True, k: int = config.CONTEXT_K) -> dict:
    own_run = TELEMETRY.run_id is None          # nested inside an agent run? reuse its id
    run = TELEMETRY.start_run("rag") if own_run else TELEMETRY.run_id
    retr = get_retriever(strategy, embedder)
    ids = retr.search(question, mode="hybrid", k=k, rerank=rerank)
    out = answer_from_context(question, ids, retr.by_id)
    out["telemetry"] = TELEMETRY.summary(run)
    if own_run:
        TELEMETRY.end_run()
    return out
