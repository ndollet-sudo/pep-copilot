"""The multi-step agent: a hand-rolled ReAct-style tool loop with explicit failure handling.

One iteration = one model call (with tools) + execution of every tool call it emits.

Termination discipline: an explicit ``final_answer`` tool AND hard bounds
(max iterations, token budget) - both, always.

Failure modes handled (each one is counted in the run result):
  F1 hallucinated tool name        -> error observation listing the real tools
  F2 invalid arguments             -> error observation with the validation message
  F3 tool raises                   -> error observation, the loop continues
  F4 empty result                  -> observation suggesting a reformulation
  F5 loop (same call repeated)     -> nudge observation, no re-execution
  F6 ungrounded final answer       -> final_answer citing a PEP never observed is
                                      rejected once ("verify or remove")
  F7 max_iterations / token budget -> graceful fallback to the plain RAG pipeline
  F8 LLM provider failure          -> fallback to RAG; if that fails too, explicit error
  F9 soft termination (plain text) -> accepted, flagged as 'text';
                                      an EMPTY reply (no text, no tool call) is not an
                                      answer: the model is nudged once to continue
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .. import config
from ..llm import LLMError
from ..telemetry import TELEMETRY
from .tools import ToolRegistry, validate_args

SYSTEM_PROMPT = """You are PEP Copilot, an assistant for an internal knowledge base of 500 Python Enhancement Proposals (PEPs).
Tools:
- search_peps: full-text hybrid search. Use it for anything about the CONTENT of PEPs.
- get_pep: exact metadata (authors, status, Python version, dates) of a PEP you know the number of.
- find_peps: filter and COUNT PEPs by status / type / Python version / title.
- calculator: any arithmetic - never compute by hand.
- final_answer: call exactly once at the end, with the answer and the PEP numbers in 'sources'.
Rules:
- Answer only from tool observations, never from memory. Cite PEP numbers.
- Multi-part questions: plan, then call tools step by step (e.g. search -> get_pep -> calculator).
- Rejected or Superseded PEPs are not the current state of Python: say so.
- If the knowledge base does not contain the answer, call final_answer saying you could not find it.
- If a tool returns an error, read it and correct your next call.
- Answer in the language of the user's question."""


@dataclass
class AgentResult:
    question: str
    answer: str | None
    sources: list[int]
    terminated_by: str            # final_answer | text | fallback_rag | error
    iterations: int
    tool_calls: int
    failures: dict[str, int]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_s: float
    trace: list[dict] = field(default_factory=list)
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assistant_dict(message) -> dict:
    """Serialise the SDK message object into a plain dict we fully control."""
    d: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        d["tool_calls"] = [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name,
                                         "arguments": c.function.arguments or "{}"}}
                           for c in message.tool_calls]
    return d


class Agent:
    def __init__(self, llm, registry: ToolRegistry, fallback=None,
                 max_iterations: int = config.MAX_ITERATIONS,
                 max_tokens: int = config.MAX_TOKENS_PER_RUN):
        self.llm, self.registry, self.fallback = llm, registry, fallback
        self.max_iterations, self.max_tokens = max_iterations, max_tokens

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

    @staticmethod
    def _step(iteration, name, args_json, obs, ok, etype) -> dict:
        return {"iteration": iteration, "tool": name, "arguments": args_json[:300],
                "observation": obs[:300], "ok": ok, "error_type": etype}


def _canonical(args_json: str) -> str:
    """Canonical form of the arguments, so {"a":1,"b":2} and {"b":2,"a":1} are one call."""
    try:
        return json.dumps(json.loads(args_json), sort_keys=True)
    except Exception:  # noqa: BLE001
        return args_json


def build_agent(strategy: str = config.DEFAULT_STRATEGY, embedder: str = "api", rerank: bool = False,
                with_fallback: bool = True, **kw) -> Agent:
    """Wire the production agent: the configured LLM + hybrid retriever + RAG fallback."""
    from ..llm import get_client
    from ..pipeline import rag_answer
    from ..retrieval import get_retriever
    from .tools import make_tools

    registry = ToolRegistry(make_tools(get_retriever(strategy, embedder), rerank=rerank))
    fb = (lambda q: rag_answer(q, strategy, embedder, rerank=True)) if with_fallback else None
    return Agent(get_client(), registry, fallback=fb, **kw)
