"""RAGAS-style faithfulness from first principles (Es et al., 2023).

  1. decompose the answer into atomic claims        (judge call 1)
  2. verify each claim against the retrieved context (judge call 2)
  3. F = |supported claims| / |claims|

Reference-free: needs the context, not a gold answer.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import get_client


class Claims(BaseModel):
    claims: list[str] = Field(default_factory=list)


class Verdicts(BaseModel):
    verdicts: list[bool] = Field(default_factory=list)


def faithfulness(answer: str, context: str) -> float | None:
    llm = get_client()
    claims = llm.chat_json([
        {"role": "system", "content": "Decompose the answer into simple, standalone factual "
                                      "claims, each independently verifiable. Ignore citation "
                                      "markers like [1]. Refusals ('I could not find...') "
                                      "contain no claim."},
        {"role": "user", "content": answer}], Claims, name="judge_decompose").claims
    if not claims:
        return None  # nothing to verify (e.g. a refusal)
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    verdicts = llm.chat_json([
        {"role": "system", "content": "For each numbered claim, answer true if it is FULLY "
                                      "supported by the passages, false otherwise. One "
                                      "verdict per claim, same order."},
        {"role": "user", "content": f"Passages:\n{context}\n\nClaims:\n{listed}"}],
        Verdicts, name="judge_verify").verdicts
    verdicts = (verdicts + [False] * len(claims))[:len(claims)]  # missing verdict = unsupported
    return sum(verdicts) / len(claims)
