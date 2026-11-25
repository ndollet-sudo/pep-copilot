# Architecture — data flow

PEP Copilot answers questions about an internal knowledge base of **500 Python Enhancement
Proposals**. Two paths share the same retrieval stack: a plain **RAG pipeline** and a
**multi-step agent** that selects tools and falls back to RAG when it fails.

```mermaid
flowchart LR
  subgraph OFFLINE["① Offline indexing (once)"]
    SRC[(python/peps<br/>748 .rst files)] -->|build_corpus.py<br/>parse header · clean RST<br/>select 500| CORPUS[(peps.jsonl<br/>500 docs + metadata)]
    CORPUS -->|chunking.py<br/>fixed · fixed_header · section| CHUNKS[(~8.5k chunks<br/>+ PEP metadata)]
    CHUNKS -->|tokenize| BM25[(BM25 index<br/>k1=1.5 b=0.75)]
    CHUNKS -->|embeddings.py<br/>text-embedding-3-small<br/>or local LSA 256d| VEC[(L2-normalised<br/>vectors · .npy cache)]
    VEC --> FAISS[(FAISS IndexFlatIP<br/>exact cosine)]
  end

  subgraph ONLINE["② Online — one question"]
    Q([User question]) --> AGENT{{Agent loop<br/>gpt-4o-mini + tools<br/>max 8 iterations · 60k-token budget}}
    AGENT -->|tool call| VALID[validate_args<br/>registry dispatch]
    VALID -->|search_peps| HYB
    VALID -->|get_pep / find_peps| META[(metadata<br/>status · version · authors)]
    VALID -->|calculator| CALC[safe AST calculator]
    VALID -->|unknown tool / bad args / tool error| ERR[error observation<br/>→ model self-corrects]
    HYB[hybrid retrieval] --> L1[BM25 top-20]
    HYB --> L2[FAISS top-20]
    L1 --> RRF[RRF fusion k=60]
    L2 --> RRF
    RRF --> RR[optional LLM rerank<br/>→ top-5]
    RR -->|observation + PEP ids| AGENT
    META -->|observation| AGENT
    CALC -->|observation| AGENT
    ERR --> AGENT
    AGENT -->|final_answer + sources<br/>grounding check| ANS([Answer + cited PEPs])
    AGENT -.->|max iterations · budget · LLM outage| FB[Fallback: RAG pipeline<br/>hybrid → rerank → cited generation]
    FB --> ANS
  end

  FAISS -.-> L2
  BM25 -.-> L1

  subgraph OBS["③ Instrumentation & evaluation"]
    TEL[telemetry.py<br/>tokens · cost · latency · tool calls · errors]
    TR[(results/traces/*.jsonl)]
    EV[eval/<br/>retrieval: P@k · Recall@k · MRR · nDCG<br/>agent: success · tool selection · hallucinated calls · cost · p50/p95<br/>RAG: faithfulness]
    TEL --> TR
    GOLD[(held-out golden sets<br/>dev 27 · test 52 · agent 24)] --> EV
  end

  AGENT -.-> TEL
  HYB -.-> TEL
  FB -.-> TEL
```

## Components

| Layer | Module | Responsibility |
|---|---|---|
| Data | `scripts/build_corpus.py`, `corpus.py` | Clone python/peps, parse RFC-822 headers, clean RST, deterministic 500-doc selection |
| Chunking | `chunking.py` | 3 strategies (fixed 200/50, fixed + contextual header, section-aware) |
| Embedding | `embeddings.py` | `text-embedding-3-small` (prod, cached on disk) / TF-IDF+SVD (offline baseline) |
| Retrieval | `retrieval.py`, `rerank.py` | BM25 ‖ FAISS → RRF → LLM rerank (falls back to RRF order on failure) |
| Generation | `generate.py`, `pipeline.py` | Cited answers `[n]`, explicit "not found", Rejected/Superseded warning |
| Agent | `agent/tools.py`, `agent/loop.py`, `agent/graph.py` | 5 tools, schema validation, failure handling F1–F9, LangGraph port |
| LLM access | `llm.py` | OpenAI SDK client (OpenAI or OpenRouter), exponential backoff, JSON mode + Pydantic repair loop |
| Instrumentation | `telemetry.py` | Every LLM/embed/tool/retrieval event: tokens, $, latency, ok/error, run id |
| Evaluation | `eval/*` | Retrieval scorecard (dev/test), agent vs RAG scorecard, faithfulness |

## Failure handling (agent)

| # | Failure | Detection | Response |
|---|---|---|---|
| F1 | Hallucinated tool name | name ∉ registry | Error observation listing real tools |
| F2 | Invalid arguments | JSON-schema validation | Error observation with the reason |
| F3 | Tool raises | `try/except` in dispatch | Error observation, loop continues |
| F4 | Empty result | 0 hits / unknown PEP | Observation suggesting a reformulation |
| F5 | Loop | same (tool, args) > 2 times | Nudge, call not re-executed |
| F6 | Ungrounded answer | `sources` ∉ PEPs observed | final_answer rejected once |
| F7 | Max iterations / token budget | counters | Fallback to RAG pipeline |
| F8 | LLM outage | `LLMError` after 5 retries (2,4,8,16 s) | Fallback to RAG, else explicit error |
| F9 | Soft stop | answer without `final_answer` | Accepted, flagged `text` |

Every row is covered by a deterministic test in `tests/test_agent_failures.py` (scripted fake LLM).
