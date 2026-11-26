"""Statistical significance of the scorecards: bootstrap confidence intervals and paired tests.

    python -m pepcopilot.eval.significance                    # test split, API embedder
    python -m pepcopilot.eval.significance --split dev --embedder lsa

Why: the golden sets are small (52 retrieval questions, 24 agent questions), so one question
moves Hit@5 by ~0.02 and the agent success rate by ~0.04. Before saying "A beats B" we check
whether the gap survives resampling the questions.

  bootstrap_ci        percentile bootstrap CI of a mean (resample questions with replacement)
  paired_bootstrap    CI + two-sided p-value of mean(A - B) on the SAME questions
                      (paired: both systems answered every question, so we resample question ids)
  mcnemar_exact       exact McNemar test for two binary outcomes on the same questions
                      (only the discordant pairs b = A ok / B ko, c = A ko / B ok carry information;
                      under H0 b ~ Binomial(b + c, 1/2))

Reads the per-question CSVs already in results/ (no API call, no cost) and writes
results/significance_<split>_<embedder>.md.
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[3] / "results"
METRICS = ["Hit", "R", "P", "MRR", "nDCG"]
METRIC_NAMES = {"Hit": "Hit@5", "R": "Recall@5", "P": "precision@5", "MRR": "MRR", "nDCG": "nDCG@5"}

# Comparisons the README discusses, as (label, variant A, variant B) with variant = (strategy, mode).
COMPARISONS = [
    ("rerank vs no rerank (fixed_header)", ("fixed_header", "hybrid+rerank"), ("fixed_header", "hybrid")),
    ("hybrid vs BM25 (fixed_header)", ("fixed_header", "hybrid"), ("fixed_header", "bm25")),
    ("hybrid vs dense (fixed_header)", ("fixed_header", "hybrid"), ("fixed_header", "dense")),
    ("fixed vs fixed_header (rerank)", ("fixed", "hybrid+rerank"), ("fixed_header", "hybrid+rerank")),
    ("section vs fixed_header (rerank)", ("section", "hybrid+rerank"), ("fixed_header", "hybrid+rerank")),
]


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) CI."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(a, b, n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float, float, float]:
    """mean(a - b), its CI, and a two-sided bootstrap p-value for H0: mean difference = 0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired_bootstrap needs two non-empty arrays of the same length")
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    if np.all(d == 0):
        return 0.0, 0.0, 0.0, 1.0
    # share of resamples on the other side of 0 (or at 0), doubled, capped at 1
    p = 2 * min(np.mean(means <= 0), np.mean(means >= 0))
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def mcnemar_exact(a, b) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired booleans. Returns (b, c, p-value)."""
    a, b_ = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("mcnemar_exact needs two arrays of the same length")
    n01 = int(np.sum(a & ~b_))   # A right, B wrong
    n10 = int(np.sum(~a & b_))   # A wrong, B right
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, float(min(1.0, p))


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def retrieval_tables(per_q: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows = []
    for (strategy, variant), g in per_q.groupby(["strategy", "variant"], sort=False):
        row = {"chunking": strategy, "retrieval": variant, "n": len(g)}
        for m in METRICS:
            row[METRIC_NAMES[m]] = _fmt_ci(*bootstrap_ci(g[m], n_boot=n_boot, seed=seed))
        ci_rows.append(row)

    cmp_rows = []
    for label, va, vb in COMPARISONS:
        ga = per_q[(per_q.strategy == va[0]) & (per_q.variant == va[1])].set_index("qid")
        gb = per_q[(per_q.strategy == vb[0]) & (per_q.variant == vb[1])].set_index("qid")
        common = ga.index.intersection(gb.index)
        if common.empty:
            continue
        for m in ["Hit", "P", "MRR"]:
            diff, lo, hi, p = paired_bootstrap(ga.loc[common, m], gb.loc[common, m],
                                               n_boot=n_boot, seed=seed)
            cmp_rows.append({"comparison (A vs B)": label, "metric": METRIC_NAMES[m],
                             "A - B": f"{diff:+.3f}", "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                             "p": round(p, 4), "significant (5%)": "yes" if hi < 0 or lo > 0 else "no"})
    return pd.DataFrame(ci_rows), pd.DataFrame(cmp_rows)


def agent_table(agent: pd.DataFrame, rag: pd.DataFrame, n_boot: int,
                seed: int) -> tuple[pd.DataFrame, dict]:
    df = agent.set_index("qid")[["success"]].join(rag.set_index("qid")[["success"]],
                                                  lsuffix="_agent", rsuffix="_rag", how="inner")
    a, r = df["success_agent"].astype(bool), df["success_rag"].astype(bool)
    n01, n10, p = mcnemar_exact(a, r)
    diff, lo, hi, _ = paired_bootstrap(a.astype(float), r.astype(float), n_boot=n_boot, seed=seed)
    return pd.DataFrame([
        {"system": "agent", "n": len(df), "success rate [95% CI]": _fmt_ci(*bootstrap_ci(a, n_boot, seed=seed))},
        {"system": "RAG-only", "n": len(df), "success rate [95% CI]": _fmt_ci(*bootstrap_ci(r, n_boot, seed=seed))},
    ]), {"agent ok / RAG ko": n01, "agent ko / RAG ok": n10, "McNemar exact p": p,
         "difference [95% CI]": f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"}


def build_report(split: str, embedder: str, n_boot: int, seed: int) -> str:
    per_q = pd.read_csv(RESULTS / f"retrieval_{split}_{embedder}_per_question.csv")
    ci, cmp = retrieval_tables(per_q, n_boot, seed)
    out = [f"# Significance - retrieval `{split}` / `{embedder}`, k=5",
           "",
           f"Percentile bootstrap over questions, {n_boot} resamples, seed {seed}. "
           "Mean [95% CI]. Paired comparisons resample the same question ids for A and B; "
           "a difference is significant at 5% when its CI excludes 0.",
           "", "## Confidence intervals per variant", "", ci.to_markdown(index=False),
           "", "## Paired comparisons", "", cmp.to_markdown(index=False)]

    agent_csv, rag_csv = RESULTS / "agent_test_runs.csv", RESULTS / "rag_test_runs.csv"
    if split == "test" and agent_csv.exists() and rag_csv.exists():
        table, mc = agent_table(pd.read_csv(agent_csv), pd.read_csv(rag_csv), n_boot, seed)
        out += ["", "## End-to-end: agent vs RAG-only (held-out agent set)", "",
                table.to_markdown(index=False), "",
                "| " + " | ".join(mc) + " |", "|" + "---|" * len(mc),
                "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in mc.values()) + " |"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--embedder", default="api", choices=["api", "lsa"])
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    report = build_report(a.split, a.embedder, a.n_boot, a.seed)
    path = RESULTS / f"significance_{a.split}_{a.embedder}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
