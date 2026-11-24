"""Retrieval scorecard: chunking strategy x retrieval mode, on a golden split.

    python -m pepcopilot.eval.retrieval_eval --split dev  --embedder lsa
    python -m pepcopilot.eval.retrieval_eval --split test --embedder api --rerank

Tune on ``dev``; report on ``test`` (held-out, never used for tuning).
Writes results/retrieval_<split>_<embedder>.md / .csv and a per-question CSV.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .. import config
from ..retrieval import get_retriever
from ..telemetry import TELEMETRY
from .metrics import (doc_ranking, hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k,
                      relevant_ids)


def load_golden(split: str) -> list[dict]:
    path = config.GOLDEN_DIR / f"retrieval_{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(split: str, embedder: str, strategies: list[str], modes: list[str],
             rerank: bool, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    golden = load_golden(split)
    rows, per_q = [], []
    variants = [(m, False) for m in modes] + ([("hybrid", True)] if rerank else [])
    for strategy in strategies:
        retr = get_retriever(strategy, embedder)
        for mode, rr in variants:
            name = f"{mode}{'+rerank' if rr else ''}"
            TELEMETRY.reset()
            lat = []
            m = {key: [] for key in ("R", "P", "MRR", "nDCG", "Hit", "docHit", "docMRR")}
            for g in golden:
                gold = relevant_ids(retr.chunks, g["gold_docs"], g["evidence"])
                t0 = time.perf_counter()
                # MRR is computed on the 20-candidate list, P/R/nDCG on the top-k context
                ranked = retr.search(g["question"], mode=mode, k=config.CANDIDATES if not rr else k,
                                     rerank=rr)
                lat.append((time.perf_counter() - t0) * 1000)
                docs = doc_ranking(ranked)
                vals = dict(R=recall_at_k(ranked, gold, k), P=precision_at_k(ranked, gold, k),
                            MRR=mrr(ranked, gold), nDCG=ndcg_at_k(ranked, gold, k),
                            Hit=hit_at_k(ranked, gold, k),
                            docHit=hit_at_k(docs, set(g["gold_docs"]), 3),
                            docMRR=mrr(docs, set(g["gold_docs"])))
                for key, v in vals.items():
                    m[key].append(v)
                per_q.append({"strategy": strategy, "variant": name, "qid": g["qid"],
                              "tag": g["tag"], "question": g["question"],
                              "top1": ranked[0] if ranked else None, **vals})
            s = TELEMETRY.summary()
            rows.append({"chunking": strategy, "retrieval": name, "n_chunks": len(retr.chunks),
                         f"Hit@{k}": np.mean(m["Hit"]), f"Recall@{k}": np.mean(m["R"]),
                         f"precision@{k}": np.mean(m["P"]), "MRR": np.mean(m["MRR"]),
                         f"nDCG@{k}": np.mean(m["nDCG"]), "doc_Hit@3": np.mean(m["docHit"]),
                         "doc_MRR": np.mean(m["docMRR"]),
                         "latency_ms_p50": np.percentile(lat, 50),
                         "latency_ms_p95": np.percentile(lat, 95),
                         "cost_usd_per_q": s["cost_usd"] / len(golden),
                         "tokens_per_q": (s["tokens_in"] + s["tokens_out"]) / len(golden)})
            print(f"  {strategy:13s} {name:15s} Hit@{k} {rows[-1][f'Hit@{k}']:.3f}  "
                  f"P@{k} {rows[-1][f'precision@{k}']:.3f}  MRR {rows[-1]['MRR']:.3f}")
    return pd.DataFrame(rows).round(4), pd.DataFrame(per_q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--embedder", default="lsa", choices=["lsa", "api"])
    ap.add_argument("--strategies", default="fixed,fixed_header,section")
    ap.add_argument("--modes", default="bm25,dense,hybrid")
    ap.add_argument("--rerank", action="store_true", help="add the hybrid+LLM-rerank variant")
    ap.add_argument("--k", type=int, default=config.CONTEXT_K)
    a = ap.parse_args()
    print(f"Retrieval eval - split={a.split} embedder={a.embedder}")
    table, per_q = evaluate(a.split, a.embedder, a.strategies.split(","), a.modes.split(","),
                            a.rerank, a.k)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    stem = config.RESULTS_DIR / f"retrieval_{a.split}_{a.embedder}"
    table.to_csv(f"{stem}.csv", index=False)
    per_q.to_csv(f"{stem}_per_question.csv", index=False)
    by_tag = per_q.groupby(["strategy", "variant", "tag"])[["Hit", "MRR"]].mean().round(3)
    with open(f"{stem}.md", "w", encoding="utf-8") as f:
        f.write(f"# Retrieval scorecard - split `{a.split}`, embedder `{a.embedder}`, k={a.k}\n\n")
        f.write(f"{len(load_golden(a.split))} questions. Relevant chunk = gold PEP AND contains "
                "the evidence pattern. MRR over the 20-candidate list.\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n\n## Hit@k and MRR by question type\n\n")
        f.write(by_tag.unstack("tag").to_markdown())
        f.write("\n")
    print(f"-> {stem}.md")


if __name__ == "__main__":
    main()
