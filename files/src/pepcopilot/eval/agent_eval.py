"""End-to-end scorecard on the held-out agent golden set: agent vs plain RAG.

    python -m pepcopilot.eval.agent_eval                    # agent only
    python -m pepcopilot.eval.agent_eval --compare-rag      # + RAG-only baseline & faithfulness

Metrics per system: success rate (answer check), source accuracy, tool-selection accuracy,
mean iterations / tool calls, hallucinated-call rate, fallback rate, tokens, cost and
latency (p50 / p95) per question. Full traces go to results/traces/.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from .. import config
from ..telemetry import TELEMETRY


def load_agent_golden() -> list[dict]:
    with open(config.GOLDEN_DIR / "agent_test.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_answer(answer: str | None, must_contain: list[str]) -> bool:
    """Every item must appear (case-insensitive); 'a|b' means a OR b.
    Numbers are matched as whole numbers (so '8' does not match '2018')."""
    text = (answer or "").lower()
    for item in must_contain:
        alts = [a.strip().lower() for a in item.split("|")]
        ok = False
        for a in alts:
            if re.fullmatch(r"\d+(\.\d+)?", a):
                ok |= bool(re.search(rf"(?<![\d.]){re.escape(a)}(?![\d]|\.\d)", text))
            else:
                ok |= a in text
        if not ok:
            return False
    return True


def tools_ok(used: list[str], expected: list[str]) -> bool:
    return all(any(alt in used for alt in e.split("|")) for e in expected)


def run_agent_suite(golden: list[dict], args) -> pd.DataFrame:
    from ..agent.loop import build_agent

    agent = build_agent(args.strategy, args.embedder, rerank=args.rerank)
    rows = []
    for g in golden:
        r = agent.run(g["question"])
        used = [s["tool"] for s in r.trace if "tool" in s]
        rows.append({
            "qid": g["qid"], "type": g["type"], "question": g["question"],
            "answer": r.answer, "success": check_answer(r.answer, g["must_contain"]),
            "source_ok": (set(g["gold_sources"]) <= set(r.sources)) if g["gold_sources"] else None,
            "tools_ok": tools_ok(used, g["expected_tools"]), "tools_used": ",".join(used),
            "terminated_by": r.terminated_by, "iterations": r.iterations,
            "tool_calls": r.tool_calls,
            "hallucinated_calls": r.failures["unknown_tool"] + r.failures["invalid_args"],
            "failures": json.dumps({k: v for k, v in r.failures.items() if v}),
            "tokens": r.tokens_in + r.tokens_out, "cost_usd": r.cost_usd,
            "latency_s": r.latency_s})
        print(f"  {g['qid']} {'OK ' if rows[-1]['success'] else 'KO '} it={r.iterations} "
              f"{r.terminated_by:12s} ${r.cost_usd:.4f} {r.latency_s:5.1f}s  {(r.answer or '')[:70]!r}")
    return pd.DataFrame(rows)


def run_rag_suite(golden: list[dict], args, with_faithfulness: bool) -> pd.DataFrame:
    from ..eval.faithfulness import faithfulness
    from ..generate import format_context
    from ..pipeline import rag_answer
    from ..retrieval import get_retriever

    retr = get_retriever(args.strategy, args.embedder)
    rows = []
    for g in golden:
        t0 = time.perf_counter()
        out = rag_answer(g["question"], args.strategy, args.embedder, rerank=True)
        lat = time.perf_counter() - t0
        f = None
        if with_faithfulness:
            f = faithfulness(out["answer"], format_context(out["context_ids"], retr.by_id))
        cited = {c["pep"] for c in out["citations"]}
        rows.append({"qid": g["qid"], "type": g["type"], "answer": out["answer"],
                     "success": check_answer(out["answer"], g["must_contain"]),
                     "source_ok": (set(g["gold_sources"]) <= cited) if g["gold_sources"] else None,
                     "faithfulness": f,
                     "tokens": out["telemetry"]["tokens_in"] + out["telemetry"]["tokens_out"],
                     "cost_usd": out["telemetry"]["cost_usd"], "latency_s": round(lat, 3)})
        print(f"  {g['qid']} {'OK ' if rows[-1]['success'] else 'KO '} rag  F={f}  "
              f"{(out['answer'] or '')[:70]!r}")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, name: str) -> dict:
    s = {"system": name, "n": len(df), "success_rate": df["success"].mean(),
         "source_accuracy": df["source_ok"].dropna().astype(float).mean()}
    if "tools_ok" in df:
        total_calls = df["tool_calls"].sum()
        s.update({"tool_selection_acc": df["tools_ok"].mean(),
                  "mean_iterations": df["iterations"].mean(),
                  "mean_tool_calls": df["tool_calls"].mean(),
                  "hallucinated_call_rate": df["hallucinated_calls"].sum() / max(total_calls, 1),
                  "fallback_rate": (df["terminated_by"] == "fallback_rag").mean(),
                  "final_answer_rate": (df["terminated_by"] == "final_answer").mean()})
    if "faithfulness" in df and df["faithfulness"].notna().any():
        s["faithfulness"] = df["faithfulness"].dropna().mean()
    s.update({"tokens_per_q": df["tokens"].mean(), "cost_usd_per_q": df["cost_usd"].mean(),
              "cost_usd_total": df["cost_usd"].sum(),
              "latency_s_p50": np.percentile(df["latency_s"], 50),
              "latency_s_p95": np.percentile(df["latency_s"], 95)})
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default=config.DEFAULT_STRATEGY)
    ap.add_argument("--embedder", default="api", choices=["api", "lsa"])
    ap.add_argument("--rerank", action="store_true", help="LLM rerank inside search_peps")
    ap.add_argument("--compare-rag", action="store_true")
    ap.add_argument("--no-faithfulness", action="store_true")
    a = ap.parse_args()
    if not config.has_api_key():
        raise SystemExit(f"{config.API_KEY_ENV} is not set - the agent needs the LLM.")
    golden = load_agent_golden()
    print(f"Agent eval - {len(golden)} held-out questions")
    TELEMETRY.reset()
    agent_df = run_agent_suite(golden, a)
    summaries = [summarise(agent_df, "agent")]
    config.RESULTS_DIR.mkdir(exist_ok=True)
    agent_df.to_csv(config.RESULTS_DIR / "agent_test_runs.csv", index=False)
    if a.compare_rag:
        print("RAG-only baseline")
        rag_df = run_rag_suite(golden, a, with_faithfulness=not a.no_faithfulness)
        rag_df.to_csv(config.RESULTS_DIR / "rag_test_runs.csv", index=False)
        summaries.append(summarise(rag_df, "rag_only"))
    table = pd.DataFrame(summaries).round(4)
    by_type = agent_df.groupby("type")["success"].mean().round(3)
    trace_path = TELEMETRY.flush()
    with open(config.RESULTS_DIR / "agent_test.md", "w", encoding="utf-8") as f:
        f.write(f"# End-to-end scorecard - {len(golden)} held-out questions\n\n")
        f.write(f"Chat model `{config.CHAT_MODEL}`, embedder `{a.embedder}`, chunking `{a.strategy}`.\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n\n## Agent success by question type\n\n")
        f.write(by_type.to_frame().to_markdown())
        f.write(f"\n\nTraces: `{trace_path.relative_to(config.ROOT)}`\n")
    print(table.T.to_string())
    print(f"-> results/agent_test.md, traces -> {trace_path}")


if __name__ == "__main__":
    main()
