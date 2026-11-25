"""LLM-as-reranker: one structured-output call scores all candidates (session 5 pattern).

Failure handling: if the call fails or the JSON is unusable, we fall back to the
fused RRF order instead of crashing - the funnel degrades, it does not break.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .llm import LLMError, get_client
from .telemetry import TELEMETRY


class ChunkScore(BaseModel):
    number: int = Field(ge=1)
    score: float = Field(ge=0, le=10)


class RerankVerdict(BaseModel):
    scores: list[ChunkScore]


RERANK_SYSTEM = (
    "You are a relevance judge inside a retrieval system over Python Enhancement Proposals "
    "(PEPs). You see one question and numbered passage candidates. Score each candidate "
    "from 0 (irrelevant) to 10 (directly answers the question). Judge relevance to the "
    "question only. Return a score for every candidate."
)


def llm_rerank(question: str, candidate_ids: list[str], by_id: dict, top: int = 5,
               max_chars: int = 600) -> list[str]:
    lines = [f"Question: {question}", "", "Candidates:"]
    for n, cid in enumerate(candidate_ids, start=1):
        c = by_id[cid]
        lines.append(f"[{n}] (PEP {c['pep']}: {c['title']}) {c['text'][:max_chars]}")
    try:
        verdict = get_client().chat_json(
            [{"role": "system", "content": RERANK_SYSTEM},
             {"role": "user", "content": "\n".join(lines)}], RerankVerdict, name="rerank")
    except LLMError as e:
        TELEMETRY.record("fallback", "rerank->rrf_order", ok=False, error=str(e)[:300])
        return candidate_ids[:top]
    score = {s.number: s.score for s in verdict.scores}
    # stable sort: ties (and missing scores) keep the RRF order
    order = sorted(range(len(candidate_ids)), key=lambda i: -score.get(i + 1, 0.0))
    return [candidate_ids[i] for i in order[:top]]
