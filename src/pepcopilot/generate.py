"""Grounded generation: answer ONLY from the retrieved passages, every claim cited [n]."""
from __future__ import annotations

import re

from .llm import get_client

SYSTEM = (
    "You answer questions about Python Enhancement Proposals (PEPs) using ONLY the passages "
    "provided. Every factual claim must cite its supporting passage as [1], [2], ... "
    "Name the PEP number when you use it. Quote identifiers, versions and numbers exactly. "
    "Beware: some passages come from Rejected or Superseded PEPs - say so when relevant. "
    "If the passages do not contain the answer, reply exactly: "
    "\"I could not find this in the PEP knowledge base.\" Never invent facts. "
    "Answer in the language of the question."
)
NOT_FOUND = "I could not find this in the PEP knowledge base."


def format_context(chunk_ids: list[str], by_id: dict) -> str:
    blocks = []
    for n, cid in enumerate(chunk_ids, start=1):
        c = by_id[cid]
        sec = f" > {c['section']}" if c.get("section") else ""
        blocks.append(f"[{n}] PEP {c['pep']}: {c['title']} ({c['status']}){sec}\n{c['text']}")
    return "\n\n".join(blocks)


def answer_from_context(question: str, chunk_ids: list[str], by_id: dict) -> dict:
    user = format_context(chunk_ids, by_id) + f"\n\nQuestion: {question}"
    r = get_client().chat([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user}], name="generate")
    text = r.choices[0].message.content or ""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(chunk_ids)})
    return {"answer": text,
            "context_ids": chunk_ids,
            "citations": [{"n": n, "chunk_id": chunk_ids[n - 1],
                           "pep": by_id[chunk_ids[n - 1]]["pep"]} for n in cited]}
