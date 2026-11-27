# Code tour — the key parts, in presentation order

Every excerpt below is copied from the repository and links to the exact lines on GitHub. Order follows the architecture slide: **Prepare → Answer → Fallback → Measure**, then the tests.

| # | Part | File | Lines |
|---|---|---|---|
| 1a | Select the 500 PEPs | [`scripts/build_corpus.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/scripts/build_corpus.py#L101-L107) | 101–107 |
| 1b | Chunking: 200-word windows, 50 overlap | [`src/pepcopilot/chunking.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/chunking.py#L25-L37) | 25–37 |
| 1c | Contextual header on every chunk | [`src/pepcopilot/chunking.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/chunking.py#L114-L119) | 114–119 |
| 1d | Build the two indexes: BM25 + FAISS | [`src/pepcopilot/retrieval.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/retrieval.py#L44-L57) | 44–57 |
| 2a | Reciprocal Rank Fusion | [`src/pepcopilot/retrieval.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/retrieval.py#L34-L40) | 34–40 |
| 2b | The search funnel | [`src/pepcopilot/retrieval.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/retrieval.py#L76-L90) | 76–90 |
| 2c | LLM reranker (with its own fallback) | [`src/pepcopilot/rerank.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/rerank.py#L31-L47) | 31–47 |
| 2d | How a tool is described to the LLM (JSON schema) | [`src/pepcopilot/agent/tools.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/tools.py#L177-L182) | 177–182 |
| 2e | The agent loop | [`src/pepcopilot/agent/loop.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/loop.py#L88-L184) | 88–184 |
| 2f | dispatch(): a tool never crashes the loop | [`src/pepcopilot/agent/tools.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/tools.py#L233-L244) | 233–244 |
| 2g | Argument validation against the schema | [`src/pepcopilot/agent/tools.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/tools.py#L41-L77) | 41–77 |
| 2h | Safe calculator (no eval) | [`src/pepcopilot/agent/tools.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/tools.py#L86-L102) | 86–102 |
| 3a | Switch to the simple RAG | [`src/pepcopilot/agent/loop.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/loop.py#L173-L181) | 173–181 |
| 4a | Every call is timed and metered | [`src/pepcopilot/telemetry.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/telemetry.py#L60-L74) | 60–74 |
| 4b | Retries with exponential backoff | [`src/pepcopilot/llm.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/llm.py#L47-L57) | 47–57 |
| 4c | Strict relevance rule for scoring | [`src/pepcopilot/eval/metrics.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/eval/metrics.py#L18-L25) | 18–25 |
| 4d | MRR | [`src/pepcopilot/eval/metrics.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/eval/metrics.py#L40-L44) | 40–44 |
| 5a | A fake LLM that plays a script | [`tests/conftest.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/tests/conftest.py#L26-L40) | 26–40 |
| 5b | Example: invented tool, then recovery (F1) | [`tests/test_agent_failures.py`](https://github.com/ndollet-sudo/pep-copilot/blob/main/tests/test_agent_failures.py#L21-L28) | 21–28 |

## 1. Prepare — built once, before any question

### 1a · Select the 500 PEPs

**File:** [`scripts/build_corpus.py` lines 101–107](https://github.com/ndollet-sudo/pep-copilot/blob/main/scripts/build_corpus.py#L101-L107)  
**What to say:** Drop April-Fool and short PEPs, rank by status, keep the first 500. Rejected/Superseded PEPs stay on purpose as traps.

```python
def select(docs: list[dict], n: int = N_DOCS) -> list[dict]:
    pool = [d for d in docs if d["status"] != "April Fool!" and d["n_words"] >= MIN_WORDS]
    rank = {s: i for i, s in enumerate(STATUS_PRIORITY)}
    pool.sort(key=lambda d: (rank.get(d["status"], 99), d["pep"]))
    chosen = pool[:n]
    chosen.sort(key=lambda d: d["pep"])
    return chosen
```

### 1b · Chunking: 200-word windows, 50 overlap

**File:** [`src/pepcopilot/chunking.py` lines 25–37](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/chunking.py#L25-L37)  
**What to say:** Each window starts 150 words after the previous one, so consecutive chunks share 50 words.

```python
def fixed_windows(text: str, c: int = config.FIXED_CHUNK_WORDS,
                  o: int = config.FIXED_OVERLAP_WORDS) -> list[str]:
    """Chunk i covers words[i*(c-o) : i*(c-o)+c]; the last window may be short."""
    if not 0 <= o < c:
        raise ValueError("overlap must satisfy 0 <= o < c")
    words = text.split()
    step, chunks, start = c - o, [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + c]))
        if start + c >= len(words):
            break
        start += step
    return chunks
```

### 1c · Contextual header on every chunk

**File:** [`src/pepcopilot/chunking.py` lines 114–119](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/chunking.py#L114-L119)  
**What to say:** Each chunk is prefixed with “PEP 572: Assignment Expressions [Final, Python 3.8]” so it knows where it comes from.

```python
def header(doc: dict, section: str | None = None) -> str:
    meta = [doc.get("status", "")]
    if doc.get("python_version"):
        meta.append(f"Python {doc['python_version']}")
    h = f"PEP {doc['pep']}: {doc['title']} [{', '.join(m for m in meta if m)}]"
    return f"{h} > {section}" if section else h
```

### 1d · Build the two indexes: BM25 + FAISS

**File:** [`src/pepcopilot/retrieval.py` lines 44–57](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/retrieval.py#L44-L57)  
**What to say:** BM25 on the tokens; FAISS IndexFlatIP on L2-normalised embeddings, i.e. exact cosine similarity.

```python
def __init__(self, chunks: list[dict], embedder: Embedder):
    self.chunks = chunks
    self.by_id = {c["chunk_id"]: c for c in chunks}
    self.embedder = embedder
    texts = [c["embed_text"] for c in chunks]
    with TELEMETRY.span("index", "build_bm25", n=len(chunks)):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts],
                              k1=config.BM25_K1, b=config.BM25_B)
    with TELEMETRY.span("index", f"build_faiss_{embedder.name}", n=len(chunks)):
        embedder.fit(texts)
        matrix = embedder.encode_docs(texts)
        self.index = faiss.IndexFlatIP(matrix.shape[1])  # exact cosine on unit vectors
        self.index.add(matrix)
    assert self.index.ntotal == len(chunks)
```

## 2. Answer — for every question

### Hybrid search

### 2a · Reciprocal Rank Fusion

**File:** [`src/pepcopilot/retrieval.py` lines 34–40](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/retrieval.py#L34-L40)  
**What to say:** Score = Σ 1/(60 + rank). It merges ranks, not scores, because BM25 and cosine scores are not on the same scale.

```python
def rrf_fuse(*ranked_lists: list[str], k: int = config.RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over ranked id lists (best first). Ties keep first-seen order."""
    scores: Counter = Counter()
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] += 1.0 / (k + rank)
    return [cid for cid, _ in scores.most_common()]
```

### 2b · The search funnel

**File:** [`src/pepcopilot/retrieval.py` lines 76–90](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/retrieval.py#L76-L90)  
**What to say:** Top-20 from BM25 and FAISS, fused, then optionally reranked by the LLM down to the top 5.

```python
def search(self, query: str, mode: str = "hybrid", k: int = config.CONTEXT_K,
           rerank: bool = False) -> list[str]:
    """Public entry point used by the pipeline, the agent tool and the evaluation."""
    if mode == "bm25":
        pool = self.bm25_search(query, max(k, config.CANDIDATES))
    elif mode == "dense":
        pool = self.dense_search(query, max(k, config.CANDIDATES))
    elif mode == "hybrid":
        pool = self.hybrid_search(query, max(k, config.CANDIDATES))
    else:
        raise ValueError(f"unknown retrieval mode {mode!r}")
    if rerank and pool:
        from .rerank import llm_rerank
        return llm_rerank(query, pool, self.by_id, top=k)
    return pool[:k]
```

### 2c · LLM reranker (with its own fallback)

**File:** [`src/pepcopilot/rerank.py` lines 31–47](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/rerank.py#L31-L47)  
**What to say:** One call scores the 20 candidates; if it fails, we keep the RRF order instead of crashing.

```python
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
```

### The agent

### 2d · How a tool is described to the LLM (JSON schema)

**File:** [`src/pepcopilot/agent/tools.py` lines 177–182](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/tools.py#L177-L182)  
**What to say:** The LLM reads the description to choose the tool, and the parameters to know which arguments to send.

```python
"get_pep": {
    "description": "Exact metadata of one PEP by number: title, authors, status, type, "
                   "creation date, target Python version, replaces/superseded-by.",
    "parameters": {"type": "object", "properties": {
        "number": {"type": "integer", "minimum": 1, "description": "PEP number, e.g. 572"}},
        "required": ["number"]}},
```

### 2e · The agent loop

**File:** [`src/pepcopilot/agent/loop.py` lines 88–184](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/loop.py#L88-L184)  
**What to say:** One iteration = one LLM call + execution of the tools it asks for. Bounded by 8 steps and 60k tokens. F5 (repeated call), F6 (unverified source), F7 (budget), F8 (API down) and F9 (empty reply) are all handled here.

```python
def run(self, question: str) -> AgentResult:
    run_id = TELEMETRY.start_run("agent")
    t0 = time.perf_counter()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question}]
    failures = {k: 0 for k in ("unknown_tool", "invalid_args", "tool_error", "empty",
                               "repeated_call", "ungrounded_answer", "budget", "llm_error",
                               "empty_reply")}
    trace: list[dict] = []
    seen_calls: dict[tuple[str, str], int] = {}
    observed_peps: set[int] = set()
    grounding_warned = False
    tok_in = tok_out = 0

    def finish(answer, sources, how, iterations):
        s = TELEMETRY.summary(run_id)
        TELEMETRY.end_run()
        return AgentResult(question, answer, sorted(set(sources or [])), how, iterations,
                           len(trace), failures, s["tokens_in"], s["tokens_out"],
                           s["cost_usd"], round(time.perf_counter() - t0, 3), trace, run_id)

    iteration = 0
    for iteration in range(1, self.max_iterations + 1):
        if tok_in + tok_out > self.max_tokens:                         # F7 budget
            failures["budget"] += 1
            break
        try:
            resp = self.llm.chat(messages, tools=self.registry.schemas, name="agent_step")
        except LLMError as e:                                          # F8 provider
            failures["llm_error"] += 1
            trace.append({"iteration": iteration, "event": "llm_error", "error": str(e)[:200]})
            break
        if getattr(resp, "usage", None):
            tok_in += resp.usage.prompt_tokens or 0
            tok_out += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        messages.append(_assistant_dict(msg))
        calls = msg.tool_calls or []
        if not calls:                                                  # F9 soft stop
            if (msg.content or "").strip() or failures["empty_reply"]:
                return finish(msg.content, sorted(observed_peps), "text", iteration)
            failures["empty_reply"] += 1          # empty turn: nudge once, keep going
            trace.append({"iteration": iteration, "event": "empty_reply_nudge"})
            messages.append({"role": "user", "content":
                             "You stopped without an answer. Continue: call the next "
                             "tool you need (e.g. calculator), then final_answer."})
            continue

        for call in calls:
            name, args_json = call.function.name, call.function.arguments or "{}"
            key = (name, _canonical(args_json))
            seen_calls[key] = seen_calls.get(key, 0) + 1

            if name == "final_answer" and self.registry.classify(name, args_json) == "ok":
                _, args, _ = validate_args(self.registry.params(name), args_json)
                sources = args.get("sources") or []
                unseen = [p for p in sources if p not in observed_peps]
                if unseen and not grounding_warned:                    # F6 grounding
                    grounding_warned = True
                    failures["ungrounded_answer"] += 1
                    obs = (f"Rejected: you cite PEP(s) {unseen} but no tool observation "
                           "showed them. Verify with get_pep/search_peps, or remove them.")
                    trace.append(self._step(iteration, name, args_json, obs, False, "ungrounded"))
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})
                    continue
                trace.append(self._step(iteration, name, args_json, "final answer", True, None))
                TELEMETRY.record("tool", name)
                return finish(args["answer"], sources, "final_answer", iteration)

            if seen_calls[key] > config.MAX_REPEATED_CALLS:                # F5 loop
                failures["repeated_call"] += 1
                obs = ("You already made this exact call. Its result will not change: "
                       "use the observations you have, try a different query, or call final_answer.")
                ok, etype = False, "repeated_call"
            else:
                with TELEMETRY.span("tool", name, arguments=args_json[:300]) as ev:
                    res = self.registry.dispatch(name, args_json)
                    ev["ok"], ev["error"] = res.ok, res.error_type
                obs, ok, etype = res.observation, res.ok, res.error_type
                observed_peps.update(res.peps)
                if etype:
                    failures[etype] += 1                                # F1-F4
            trace.append(self._step(iteration, name, args_json, obs, ok, etype))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})

    # F7/F8: graceful degradation - the plain RAG pipeline answers instead.
    if self.fallback is not None:
        try:
            fb = self.fallback(question)
            trace.append({"iteration": iteration, "event": "fallback_rag"})
            return finish(fb["answer"], [c["pep"] for c in fb.get("citations", [])],
                          "fallback_rag", iteration)
        except Exception as e:  # noqa: BLE001
            trace.append({"iteration": iteration, "event": "fallback_failed", "error": str(e)[:200]})
    return finish("Sorry - I could not complete this request (agent stopped: "
                  f"{', '.join(k for k, v in failures.items() if v) or 'max iterations'}).",
                  [], "error", iteration)
```

### 2f · dispatch(): a tool never crashes the loop

**File:** [`src/pepcopilot/agent/tools.py` lines 233–244](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/tools.py#L233-L244)  
**What to say:** Unknown tool (F1), invalid arguments (F2) and tool exceptions (F3) all become a message the model can read.

```python
def dispatch(self, name: str, arguments_json: str) -> ToolResult:
    if name not in self.tools:
        return ToolResult(False, f"Error: unknown tool '{name}'. Available tools: "
                                 f"{', '.join(sorted(self.tools))}.", "unknown_tool")
    ok, args, err = validate_args(self.params(name), arguments_json)
    if not ok:
        return ToolResult(False, f"Error: invalid arguments for tool '{name}': {err}",
                          "invalid_args")
    try:
        return self.tools[name][1](**args)
    except Exception as e:  # noqa: BLE001 - a failing tool is an observation, not a crash
        return ToolResult(False, f"Error: tool '{name}' failed: {e}", "tool_error")
```

### 2g · Argument validation against the schema

**File:** [`src/pepcopilot/agent/tools.py` lines 41–77](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/tools.py#L41-L77)  
**What to say:** Required fields, types, bounds and allowed values are checked before anything runs.

```python
def validate_args(schema: dict, arguments_json: str) -> tuple[bool, dict | None, str]:
    """Check raw JSON arguments against a (small) JSON-Schema subset. Never raises.
    Returns (ok, filtered_args, error_message_for_the_model)."""
    try:
        parsed = json.loads(arguments_json or "{}")
    except Exception as e:  # noqa: BLE001
        return False, None, f"arguments must be valid JSON ({e})"
    if not isinstance(parsed, dict):
        return False, None, "arguments must be a JSON object mapping field names to values"
    props = schema.get("properties", {})
    missing = [f for f in schema.get("required", []) if f not in parsed]
    if missing:
        return False, None, f"missing required field(s): {', '.join(missing)}"
    args = {}
    for field, value in parsed.items():
        if field not in props:
            continue                                   # unknown fields filtered, not fatal
        spec = props[field]
        t = spec.get("type")
        py = _JSON_TYPES.get(t)
        if t == "integer" and isinstance(value, str) and value.strip().isdigit():
            value = int(value)                         # forgiving coercion: "572" -> 572
        if py and (not isinstance(value, py) or (t in ("integer", "number") and isinstance(value, bool))):
            return False, None, f"field '{field}' must be of type {t}"
        if "enum" in spec and value not in spec["enum"]:
            return False, None, f"field '{field}' must be one of {spec['enum']}"
        if "minimum" in spec and value < spec["minimum"]:
            return False, None, f"field '{field}' must be >= {spec['minimum']}"
        if "maximum" in spec and value > spec["maximum"]:
            return False, None, f"field '{field}' must be <= {spec['maximum']}"
        if t == "array" and spec.get("items", {}).get("type") == "integer":
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                return False, None, f"field '{field}' must be a list of integers"
        if t == "string" and not value.strip():
            return False, None, f"field '{field}' must not be empty"
        args[field] = value
    return True, args, ""
```

### 2h · Safe calculator (no eval)

**File:** [`src/pepcopilot/agent/tools.py` lines 86–102](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/tools.py#L86-L102)  
**What to say:** The expression is parsed into a syntax tree and only arithmetic nodes are allowed, so no code can run.

```python
def calculator(expression: str) -> ToolResult:
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression: use numbers and + - * / ** % // ( ); "
                         "for powers use ** not ^")
    result = ev(ast.parse(expression, mode="eval").body)
    if isinstance(result, float):
        result = int(result) if result.is_integer() else round(result, 10)
    return ToolResult(True, str(result))
```

## 3. Fallback — the backup plan

### 3a · Switch to the simple RAG

**File:** [`src/pepcopilot/agent/loop.py` lines 173–181](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/agent/loop.py#L173-L181)  
**What to say:** If the loop ends without an answer (limit reached or API down), the plain RAG pipeline answers instead: one search, one cited answer, same indexes.

```python
# F7/F8: graceful degradation - the plain RAG pipeline answers instead.
if self.fallback is not None:
    try:
        fb = self.fallback(question)
        trace.append({"iteration": iteration, "event": "fallback_rag"})
        return finish(fb["answer"], [c["pep"] for c in fb.get("citations", [])],
                      "fallback_rag", iteration)
    except Exception as e:  # noqa: BLE001
        trace.append({"iteration": iteration, "event": "fallback_failed", "error": str(e)[:200]})
```

## 4. Measure — cost, latency, scores

### 4a · Every call is timed and metered

**File:** [`src/pepcopilot/telemetry.py` lines 60–74](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/telemetry.py#L60-L74)  
**What to say:** Records latency, tokens, cost and errors for each LLM call, embedding, search and tool call.

```python
@contextmanager
def span(self, kind: str, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block; the yielded dict can be enriched (tokens, args, ...)."""
    ev = self.record(kind, name, **fields)
    t0 = time.perf_counter()
    try:
        yield ev
    except Exception as e:  # recorded, then re-raised: telemetry never swallows errors
        ev["ok"], ev["error"] = False, f"{type(e).__name__}: {e}"
        raise
    finally:
        ev["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if ev["kind"] in ("llm", "embed") and not ev.get("cost_usd"):
            ev["cost_usd"] = usd_for("embed" if ev["kind"] == "embed" else "chat",
                                     ev["tokens_in"], ev["tokens_out"])
```

### 4b · Retries with exponential backoff

**File:** [`src/pepcopilot/llm.py` lines 47–57](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/llm.py#L47-L57)  
**What to say:** Up to 5 attempts, waiting 2, 4, 8, 16 s; permanent errors (4xx) fail fast.

```python
def _with_retry(self, fn, what: str):
    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - provider SDK raises many types
            if attempt == config.MAX_ATTEMPTS or _status(e) in config.PERMANENT_HTTP_ERRORS:
                raise LLMError(f"{what} failed after {attempt} attempt(s): {e}") from e
            wait = config.BACKOFF_BASE_S ** attempt
            TELEMETRY.record("retry", what, attempt=attempt, wait_s=wait,
                             ok=False, error=f"{type(e).__name__}: {e}")
            time.sleep(wait)
```

### 4c · Strict relevance rule for scoring

**File:** [`src/pepcopilot/eval/metrics.py` lines 18–25](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/eval/metrics.py#L18-L25)  
**What to say:** A chunk counts as relevant only if it comes from the gold PEP AND contains the evidence.

```python
def relevant_ids(chunks: list[dict], gold_docs: list[str], evidence: str | None) -> set[str]:
    pat = re.compile(evidence, re.I) if evidence else None
    gold = set(gold_docs)
    rel = {c["chunk_id"] for c in chunks
           if c["doc_id"] in gold and (pat is None or pat.search(c["text"]))}
    if not rel:  # evidence split across a boundary: fall back to document-level gold
        rel = {c["chunk_id"] for c in chunks if c["doc_id"] in gold}
    return rel
```

### 4d · MRR

**File:** [`src/pepcopilot/eval/metrics.py` lines 40–44](https://github.com/ndollet-sudo/pep-copilot/blob/main/src/pepcopilot/eval/metrics.py#L40-L44)  
**What to say:** 1 / rank of the first relevant chunk: 1 if first, 0.5 if second, 0.25 if fourth.

```python
def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0
```

## 5. Tests — failures replayed without network

### 5a · A fake LLM that plays a script

**File:** [`tests/conftest.py` lines 26–40](https://github.com/ndollet-sudo/pep-copilot/blob/main/tests/conftest.py#L26-L40)  
**What to say:** The agent receives its LLM as a parameter, so tests give it a fake one that replays written answers.

```python
class FakeLLM:
    """Replays a script of responses; each item is a response or an Exception to raise."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, messages, tools=None, name="chat", **kw):
        from pepcopilot.telemetry import TELEMETRY
        self.calls.append(messages)
        item = self.script.pop(0) if self.script else response(content="(script exhausted)")
        if isinstance(item, Exception):
            raise item
        TELEMETRY.record("llm", name, tokens_in=item.usage.prompt_tokens,
                         tokens_out=item.usage.completion_tokens)
        return item
```

### 5b · Example: invented tool, then recovery (F1)

**File:** [`tests/test_agent_failures.py` lines 21–28](https://github.com/ndollet-sudo/pep-copilot/blob/main/tests/test_agent_failures.py#L21-L28)  
**What to say:** The fake LLM misspells a tool; the test checks the failure is counted and the agent still finishes.

```python
def test_hallucinated_tool_then_recovery(registry):
    llm = FakeLLM([response([tool_call("serach_peps", {"query": "walrus"})]),
                   response([tool_call("search_peps", {"query": "walrus"})]),
                   response([tool_call("final_answer", {"answer": "ok", "sources": [572]})])])
    r = Agent(llm, registry).run("q")
    assert r.failures["unknown_tool"] == 1 and r.terminated_by == "final_answer"
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert "Available tools" in tool_msgs[0]["content"]          # the model was told what exists
```

---

Run everything offline: `make test`. Ask a question: `make ask Q="Which PEP introduced except*?"`.
