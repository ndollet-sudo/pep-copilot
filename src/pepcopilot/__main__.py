"""Command line.

    python -m pepcopilot search "how to build without the GIL" --embedder lsa   # offline
    python -m pepcopilot ask "Which PEP introduced except* and for which version?"
    python -m pepcopilot ask "..." --rag          # plain RAG pipeline, no agent
"""
from __future__ import annotations

import argparse
import json

from . import config
from .telemetry import TELEMETRY


def main() -> None:
    ap = argparse.ArgumentParser(prog="pepcopilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="retrieval only (works offline with --embedder lsa)")
    s.add_argument("query")
    s.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    s.add_argument("--k", type=int, default=5)
    a = sub.add_parser("ask", help="answer a question (needs OPENAI_API_KEY or OPENROUTER_API_KEY)")
    a.add_argument("question")
    a.add_argument("--rag", action="store_true", help="plain RAG pipeline instead of the agent")
    for p in (s, a):
        p.add_argument("--embedder", default="api", choices=["api", "lsa"])
        p.add_argument("--strategy", default=config.DEFAULT_STRATEGY, choices=["fixed", "fixed_header", "section"])
    args = ap.parse_args()

    if args.cmd == "search":
        from .retrieval import get_retriever
        retr = get_retriever(args.strategy, args.embedder)
        for i, cid in enumerate(retr.search(args.query, mode=args.mode, k=args.k), start=1):
            c = retr.by_id[cid]
            print(f"{i}. PEP {c['pep']} - {c['title']} > {c['section']}  [{cid}]\n   {c['text'][:220]}...")
        TELEMETRY.report("search")
        return

    if args.rag:
        from .pipeline import rag_answer
        out = rag_answer(args.question, args.strategy, args.embedder)
        print(out["answer"])
        print("\nSources:", ", ".join(f"[{c['n']}] PEP {c['pep']}" for c in out["citations"]))
        print("Telemetry:", json.dumps(out["telemetry"]))
        return

    from .agent.loop import build_agent
    r = build_agent(args.strategy, args.embedder).run(args.question)
    for step in r.trace:
        if "tool" in step:
            flag = "OK " if step["ok"] else "ERR"
            print(f"  [{step['iteration']}] {flag} {step['tool']}({step['arguments'][:70]})")
        else:
            print(f"  [{step['iteration']}] {step['event']}")
    print(f"\n{r.answer}\n\nSources: {', '.join(f'PEP {p}' for p in r.sources) or '-'}")
    print(f"terminated_by={r.terminated_by} iterations={r.iterations} tool_calls={r.tool_calls} "
          f"tokens={r.tokens_in}+{r.tokens_out} cost=${r.cost_usd:.4f} latency={r.latency_s}s")
    TELEMETRY.flush()


if __name__ == "__main__":
    main()
