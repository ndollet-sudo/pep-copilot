# PEP Copilot — RAG & agentic assistant over 500 Python Enhancement Proposals

> *"Why was the walrus operator added, in which version, and who wrote the PEP?"*
> A production-grade RAG + agent system for an internal knowledge base: the design
> decisions of the Python language (PEPs = Python's architecture decision records).

**Course:** Generative AI — final project · **Team:** @ndollet-sudo (data & evaluation) · @lniat-coder (retrieval) · @stawamba-commits (LLM, generation & instrumentation) · @nayekanoute (agent) · @icetrucklr (statistical significance & CI)
**Defence:** 27 November · **Rule we follow:** nothing is "working" until it is scored on a
held-out golden set with cost and latency tracked.

## 1. The knowledge base — 500 documents

| | |
|---|---|
| Source | [`github.com/python/peps`](https://github.com/python/peps) (official PEP sources, reStructuredText) |
| Selection | deterministic (`scripts/build_corpus.py`): drop April-Fool and < 300-word PEPs, rank by status (Final > Active > Accepted > Provisional > Superseded > Rejected), keep 500 |
| Size | 500 docs · 1.27 M words · 371 Final, 60 Rejected, 25 Superseded, 32 Active… |
| Metadata | number, title, authors, status, type, created, Python version, replaces / superseded-by |
| Why it is hard | near-duplicates by design: **Rejected / Superseded PEPs** (e.g. PEP 622 → 634) are kept as hard negatives; exact identifiers (`--disable-gil`, `__fspath__`) **and** paraphrases; French questions over an English corpus |

## 2. Architecture

Full data-flow diagram and failure-handling table: **[docs/architecture.md](docs/architecture.md)**.

```
question ─► Agent (Mistral Small 4, tool calling, ≤8 iterations, 60k-token budget)
              ├─ search_peps ─► BM25 top-20 ‖ FAISS top-20 ─► RRF (k=60) ─► [LLM rerank] ─► top-5
              ├─ get_pep / find_peps ─► exact metadata, counts
              ├─ calculator ─► safe AST arithmetic
              └─ final_answer(answer, sources) ─► grounding check ─► answer + cited PEPs
            fallback on failure ─► plain RAG pipeline (hybrid ─► rerank ─► cited generation)
telemetry: tokens · $ · latency · tool calls · errors for every step ─► results/traces/
```

## 3. Quick start

**Environment** (Python ≥ 3.10):

```bash
# macOS / Linux
make setup                      # creates .venv, installs requirements.txt, creates .env
# Windows (PowerShell)
python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt; copy .env.example .env
```
Then put your key in `.env`; it is loaded automatically and git-ignored. The provider is picked from the key:
`OPENROUTER_API_KEY` → OpenRouter (Mistral Small 4 + mistral-embed, as in the labs — **used for all reported results**);
`OPENAI_API_KEY` → OpenAI (`gpt-4o-mini`, `text-embedding-3-small`). Override with `PEPCOPILOT_*` variables.

```bash
make test                       # 37 tests, offline (incl. 11 failure-injection scenarios)
python -m pepcopilot.eval.significance   # bootstrap CIs + paired tests from the per-question CSVs (free)
make eval-offline               # retrieval scorecard with the free local embedder

source .venv/bin/activate && export PYTHONPATH=src     # Windows: $env:PYTHONPATH="src"
python -m pepcopilot ask "Which PEP introduced except*, and for which Python version?"
python -m pepcopilot ask "How many Final PEPs target Python 3.11?"
python -m pepcopilot ask "Quelle PEP a introduit les f-strings ?" --rag
make eval-retrieval             # API embedder + rerank, dev and test    (~$0.5)
make eval-agent                 # agent vs RAG-only, held-out agent set  (~$0.05)
```

## 4. Evaluation protocol

| Golden set | Size | Use |
|---|---|---|
| `data/golden/retrieval_dev.jsonl` | 27 | **tuning only** (chunk size, strategy choice) |
| `data/golden/retrieval_test.jsonl` | 52 | **held-out**, reported, never used for tuning |
| `data/golden/agent_test.jsonl` | 24 | held-out end-to-end: RAG, metadata, counting, multi-step, multi-hop, unanswerable, French |

*Relevance rule:* a chunk is relevant iff it belongs to a gold PEP **and** contains the evidence
pattern (the fact that answers the question) — every evidence string was checked against the corpus.
Question tags: `fact`, `identifier` (BM25 territory), `paraphrase` (dense territory),
`trap-superseded`, `fr` (cross-lingual).

### 4.1 Retrieval — `mistral-embed`, k = 5

**Model selection on DEV (27 q).** The LLM reranker wins for every chunking strategy; among the
reranked variants, `fixed_header` has the best MRR (0.963) with Hit@5 = 1.00 → **chosen configuration:
fixed + contextual header · hybrid BM25 ‖ FAISS · RRF · LLM rerank**.

**Reported on the held-out TEST set (52 q)** — `results/retrieval_test_api.md`:

| chunking | retrieval | Hit@5 | Recall@5 | **precision@5** | **MRR** | nDCG@5 | p50 latency | $ / query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed + header | BM25 | 0.846 | 0.267 | 0.435 | 0.694 | 0.514 | 26 ms | 0 |
| fixed + header | dense (FAISS) | 0.885 | 0.319 | 0.527 | 0.689 | 0.573 | 289 ms | ~0 |
| fixed + header | hybrid RRF | 0.885 | 0.315 | 0.473 | 0.698 | 0.557 | 307 ms | ~0 |
| **fixed + header** | **hybrid + rerank** ✔ | **0.942** | **0.374** | **0.585** | **0.859** | **0.710** | 2.0 s | 0.0007 |
| fixed 200/50 | hybrid + rerank | 0.962 | 0.388 | 0.639 | 0.889 | 0.762 | 2.1 s | 0.0007 |
| section-aware | hybrid + rerank | 0.942 | 0.385 | 0.558 | 0.875 | 0.712 | 2.2 s | 0.0006 |

**What we learned**
- **The reranker is the biggest lever:** precision@5 0.473 → 0.585 and MRR 0.698 → 0.859 (+23 %)
  for 0.07 ¢ and ~1.7 s per query. Wide-and-cheap (hybrid top-20) then deep-and-few (rerank top-5) works.
- **Hybrid > each leg on recall:** BM25 alone misses every French question (Hit@5 = 0 on `fr`),
  dense alone is weaker on exact identifiers; RRF gets both (Hit@5 0.79 BM25 → 0.94 hybrid, `fixed`).
- **Multilingual embeddings matter:** French questions over an English corpus — BM25 0 %, `mistral-embed` 100 % Hit@5.
- **Contextual headers helped the weak offline embedder a lot (§4.3) but not `mistral-embed`:** on test,
  plain `fixed` + rerank is slightly ahead of our dev-chosen `fixed_header`. We **keep the dev choice**
  (changing it after seeing test would be tuning on the test set); with 52 questions the gap
  (Hit@5 0.962 vs 0.942 = one question) is within noise.
- **Superseded trap (PEP 622 vs 634):** one test question only — the rerank variants disagree on it; too small to conclude.

### 4.2 End-to-end — agent vs RAG-only, held-out agent set (24 q)

`results/agent_test.md` — chat model `mistralai/mistral-small-2603` (Mistral Small 4), embedder `mistral-embed`.

| | **Agent** | RAG-only |
|---|---:|---:|
| Success rate | **91.7 %** (22/24) | 45.8 % (11/24) |
| Source accuracy (gold PEP cited) | **100 %** | 70.6 % |
| Tool-selection accuracy | 95.8 % | — |
| Hallucinated tool calls | 0 % | — |
| Fallback to RAG | 0 % | — |
| Mean iterations / tool calls | 2.7 / 2.9 | 1 retrieval + 1 generation |
| Faithfulness (RAGAS-style) | — | 0.715 |
| Tokens / question | 4 284 | 5 375 |
| **Cost / question** | **$0.0007** | $0.0009 |
| Latency p50 / p95 | **2.1 s / 3.9 s** | 3.0 s / 4.2 s |
| Total cost of the 24-question run | $0.016 | $0.023 |

**Reading**
- The agent doubles the success rate **for less money**: RAG-only fails exactly where tools are needed —
  counting (`find_peps`), arithmetic (`calculator`) and metadata (`get_pep`) — and answers
  "I could not find this". Both systems correctly refuse the 2 unanswerable questions.
- RAG-only faithfulness 0.72: e.g. on a12 it computed "7 characters apart" itself, a claim not in the
  passages (F = 0) — the agent routes that step to the calculator instead.
- **The two agent failures (error analysis):**
  - **a18** (multi-step): `get_pep(3333)`, `get_pep(703)`, then an **empty reply** that the loop accepted
    as a soft stop → no answer. **Fix:** an empty turn is now nudged once (F9, regression test
    `test_empty_reply_is_nudged_not_accepted`). The table above is the original held-out run; we did
    not re-score on the same set after fixing.
  - **a23** (French + count): `find_peps` → `calculator` → answered **12** Process PEPs instead of 36 —
    a wrong tool argument / misread observation, not a retrieval problem. Next step: return the
    count as a structured field and assert it in the final answer.

### 4.3 Offline baseline (free, no API) — TF-IDF + SVD embedder

Used for CI and development before spending API credit (`make eval-offline`). Best on test:
fixed + header · hybrid RRF — Hit@5 0.865, precision@5 0.469, MRR 0.666; French BM25 = 0. Here the
contextual header was the biggest win (precision@5 0.381 → 0.469). Full tables: `results/retrieval_*_lsa.md`.

### 4.4 Statistical significance — are the gaps real?

`results/significance_test_api.md` (`python -m pepcopilot.eval.significance`, no API call): percentile
bootstrap 95 % CIs per variant (10 000 resamples of the questions), **paired** bootstrap on the same
question ids for A − B, and an **exact McNemar** test for agent vs RAG-only (binary success, same questions).

| comparison (held-out test) | metric | A − B [95 % CI] | p | significant |
|---|---|---:|---:|:---:|
| **agent vs RAG-only** (24 q) | success | +0.458 [+0.250, +0.667] | 0.001 (McNemar, 11 vs 0 discordant) | ✔ |
| rerank vs no rerank (fixed_header) | precision@5 | +0.112 [+0.062, +0.169] | < 0.001 | ✔ |
| rerank vs no rerank (fixed_header) | MRR | +0.161 [+0.081, +0.242] | < 0.001 | ✔ |
| rerank vs no rerank (fixed_header) | Hit@5 | +0.058 [+0.000, +0.135] | 0.09 | ✘ |
| hybrid vs BM25 (fixed_header) | MRR | +0.004 [−0.073, +0.082] | 0.92 | ✘ |
| fixed vs fixed_header (rerank) | Hit@5 | +0.019 [+0.000, +0.058] | 0.74 | ✘ |
| fixed vs fixed_header (rerank) | precision@5 | +0.054 [+0.012, +0.096] | 0.006 | ✔ (not after Bonferroni) |

**Reading**
- The two headline claims hold: the **agent beats RAG-only** and the **reranker improves ranking**
  (precision@5, MRR). Hit@5 saturates (0.88 without rerank), so the reranker's value is moving the
  right chunk up, not finding it.
- **Hybrid vs a single leg is not significant on average** over 52 questions; its value is per tag
  (BM25 = 0 on French, dense weaker on identifiers) — it avoids the catastrophic failures of each leg.
- `fixed` is ahead of the dev-chosen `fixed_header` on precision@5 (p = 0.006, but 15 comparisons:
  Bonferroni threshold 0.0033). We keep the dev choice — re-selecting on test would be tuning on test;
  re-running model selection on a larger dev set is the next step.
- Limits: questions are not independent (several share a PEP); with n = 24 the agent CI is approximate.

## 5. Instrumentation

`telemetry.py` records one event per LLM call, embedding batch, retrieval leg and tool call:
`run_id, kind, name, latency_ms, tokens_in, tokens_out, cost_usd, ok, error`.
`TELEMETRY.summary(run_id)` gives per-question totals and p50/p95 latency; traces are flushed to
`results/traces/*.jsonl`. Prices (config.py, overridable in `.env`): Mistral Small 4 $0.15 / $0.60 per M
tokens, mistral-embed $0.10 per M (OpenRouter, Mistral provider pinned so the meter matches the bill).
Whole evaluation campaign (corpus embedding × 3 strategies + 79 retrieval queries × 4 variants + 48 end-to-end runs): well under $1.

## 6. Repository map

```
scripts/build_corpus.py        corpus construction (500 PEPs)
src/pepcopilot/
  config.py  telemetry.py  llm.py            settings · instrumentation · provider access
  corpus.py  chunking.py  embeddings.py      data · 3 chunking strategies · 2 embedders
  retrieval.py  rerank.py                    BM25 ‖ FAISS → RRF → LLM rerank
  generate.py  pipeline.py                   cited generation · plain RAG
  agent/tools.py  agent/loop.py  agent/graph.py   tools · agent loop + failure handling · LangGraph
  eval/metrics.py  eval/retrieval_eval.py  eval/agent_eval.py  eval/faithfulness.py
  eval/significance.py                        bootstrap CIs · paired bootstrap · McNemar
tests/                         unit tests + failure-injection suite (no network)
data/corpus/  data/golden/     knowledge base · golden sets
results/                       scorecards (md + csv) · significance report
.github/workflows/tests.yml    CI: offline test suite on every push (Python 3.10, 3.12)
docs/                          architecture diagram
```

## 7. Limitations & next steps

- FAISS `IndexFlatIP` is exact and in-memory — fine for ~9k chunks; at scale: IVF/HNSW or pgvector
  (persistence, metadata filters, hybrid search in one query).
- The LLM reranker costs one call per query; a dedicated cross-encoder (e.g. Cohere Rerank) is cheaper.
- Golden sets are small (79 + 24); confidence intervals are wide (§4.4): most retrieval gaps < 0.05
  are not significant, the agent-vs-RAG and reranker gains are.
- Metadata questions are answered by tools, not retrieval: the agent's value is tool **selection**.
