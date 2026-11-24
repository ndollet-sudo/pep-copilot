# PEP Copilot — RAG & agentic assistant over 500 Python Enhancement Proposals

> *"Why was the walrus operator added, in which version, and who wrote the PEP?"*
> A production-grade RAG + agent system for an internal knowledge base: the design
> decisions of the Python language (PEPs = Python's architecture decision records).

**Course:** Generative AI — final project · **Team:** Member A · Member B · Member C · Member D
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
question ─► Agent (gpt-4o-mini, tool calling, ≤8 iterations, 60k-token budget)
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
Then put your key in `.env` (`OPENAI_API_KEY=sk-...`); it is loaded automatically and git-ignored.
The provider is picked from the key: `OPENAI_API_KEY` → OpenAI (`gpt-4o-mini`, `text-embedding-3-small`);
`OPENROUTER_API_KEY` → OpenRouter (Mistral, as in the labs). Override with `PEPCOPILOT_*` variables.

```bash
make test                       # 27 tests, offline (incl. 10 failure-injection scenarios)
make eval-offline               # retrieval scorecard with the free local embedder

source .venv/bin/activate && export PYTHONPATH=src     # Windows: $env:PYTHONPATH="src"
python -m pepcopilot ask "Which PEP introduced except*, and for which Python version?"
python -m pepcopilot ask "How many Final PEPs target Python 3.11?"
python -m pepcopilot ask "Quelle PEP a introduit les f-strings ?" --rag
make eval-retrieval             # API embedder + rerank, dev and test    (~$0.2)
make eval-agent                 # agent vs RAG-only, held-out agent set  (~$0.2)
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

### 4.1 Retrieval — held-out test set (52 q), offline LSA embedder, k = 5

| chunking | retrieval | chunks | Hit@5 | Recall@5 | **precision@5** | **MRR** | nDCG@5 | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed 200/50 | BM25 | 8531 | 0.788 | 0.236 | 0.369 | **0.680** | 0.467 | 11 |
| fixed 200/50 | dense | 8531 | 0.750 | 0.222 | 0.369 | 0.582 | 0.438 | 112 |
| fixed 200/50 | hybrid RRF | 8531 | 0.788 | 0.240 | 0.381 | 0.643 | 0.453 | 127 |
| fixed + header | BM25 | 8531 | 0.846 | 0.267 | 0.435 | **0.694** | 0.514 | 11 |
| fixed + header | dense | 8531 | 0.788 | 0.265 | 0.442 | 0.643 | 0.510 | 113 |
| **fixed + header** | **hybrid RRF** | 8531 | **0.865** | 0.277 | **0.469** | 0.666 | **0.532** | 126 |
| section-aware | BM25 | 9069 | 0.750 | 0.234 | 0.377 | 0.671 | 0.474 | 11 |
| section-aware | dense | 9069 | 0.788 | 0.245 | 0.415 | 0.575 | 0.454 | 65 |
| section-aware | hybrid RRF | 9069 | 0.846 | 0.296 | 0.438 | 0.631 | 0.510 | 78 |

Full tables (dev + test, per question type): `results/retrieval_{dev,test}_lsa.md`.

**What we learned**
- **The contextual header is the biggest single win** (PEP number + title + status prepended to
  each chunk before indexing): precision@5 0.381 → 0.469 for hybrid. It was chosen on *dev*
  (Hit@5 0.889 vs 0.852 for section-aware) and confirmed on *test* → default `fixed_header`.
- **Hybrid > either leg on Hit@5**, but BM25 alone keeps the best MRR on identifiers — RRF
  trades a little first-rank precision for coverage.
- **Section-aware chunking only helped after** dropping boilerplate sections (Copyright,
  References) and merging tiny sections: dense Hit@5 on dev went 0.63 → 0.81.
- **French questions: BM25 = 0 %** (no shared tokens). This is the argument for a multilingual
  dense embedder (`text-embedding-3-small`) — the API run (`make eval-retrieval`) quantifies it.
- Offline LSA is a weak, free dense baseline (no semantics across languages); production numbers
  with `text-embedding-3-small` + LLM rerank go in `results/retrieval_test_api.md`.

### 4.2 End-to-end agent vs RAG-only — held-out agent set (24 q)

Produced by `make eval-agent` → `results/agent_test.md` (success rate, source accuracy,
tool-selection accuracy, hallucinated-tool-call rate, fallback rate, faithfulness, tokens,
cost/question, latency p50/p95). *To be filled with the API run before the defence.*

## 5. Instrumentation

`telemetry.py` records one event per LLM call, embedding batch, retrieval leg and tool call:
`run_id, kind, name, latency_ms, tokens_in, tokens_out, cost_usd, ok, error`.
`TELEMETRY.summary(run_id)` gives per-question totals and p50/p95 latency; traces are flushed to
`results/traces/*.jsonl`. Prices (config.py, overridable in `.env`): gpt-4o-mini $0.15 / $0.60 per M tokens,
text-embedding-3-small $0.02 per M — check them against openai.com/api/pricing before the defence.

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
tests/                         unit tests + failure-injection suite (no network)
data/corpus/  data/golden/     knowledge base · golden sets
results/                       scorecards (md + csv)
docs/                          architecture diagram
```

## 7. Limitations & next steps

- FAISS `IndexFlatIP` is exact and in-memory — fine for ~9k chunks; at scale: IVF/HNSW or pgvector
  (persistence, metadata filters, hybrid search in one query).
- The LLM reranker costs one call per query; a dedicated cross-encoder (e.g. Cohere Rerank) is cheaper.
- Golden sets are small (79 + 24); confidence intervals are wide — differences < 0.05 are not significant.
- Metadata questions are answered by tools, not retrieval: the agent's value is tool **selection**.
