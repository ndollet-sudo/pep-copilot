"""Instrumentation: token usage, latency, cost and tool calls for every step.

Every LLM call, embedding call, retrieval and tool execution is recorded as an
event. A run (one question) groups events under a ``run_id``. Events are kept in
memory and can be flushed to JSONL (results/traces/) for later analysis.

    tel = Telemetry()
    with tel.span("retrieval", "hybrid") as ev:
        ...
        ev["n_results"] = 20
    tel.summary()  # totals: tokens, cost, latency p50/p95, tool calls, errors
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import config


def usd_for(kind: str, tokens_in: int, tokens_out: int = 0) -> float:
    if kind == "embed":
        return tokens_in * config.PRICES["embed_in"] / 1e6
    return (tokens_in * config.PRICES["chat_in"] + tokens_out * config.PRICES["chat_out"]) / 1e6


class Telemetry:
    """In-memory event log. One instance per process (see ``TELEMETRY``)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.run_id: str | None = None

    # ----- runs -----
    def start_run(self, label: str = "") -> str:
        self.run_id = f"{label + '-' if label else ''}{uuid.uuid4().hex[:8]}"
        return self.run_id

    def end_run(self) -> None:
        self.run_id = None

    # ----- events -----
    def record(self, kind: str, name: str, **fields: Any) -> dict[str, Any]:
        ev = {"ts": time.time(), "run_id": self.run_id, "kind": kind, "name": name,
              "latency_ms": 0.0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
              "ok": True, "error": None}
        ev.update(fields)
        if kind in ("llm", "embed") and not fields.get("cost_usd"):
            ev["cost_usd"] = usd_for("embed" if kind == "embed" else "chat",
                                     ev["tokens_in"], ev["tokens_out"])
        self.events.append(ev)
        return ev

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

    # ----- aggregation -----
    def select(self, run_id: str | None = None) -> list[dict[str, Any]]:
        return [e for e in self.events if run_id is None or e["run_id"] == run_id]

    def summary(self, run_id: str | None = None) -> dict[str, Any]:
        evs = self.select(run_id)
        llm = [e for e in evs if e["kind"] == "llm"]
        tools = [e for e in evs if e["kind"] == "tool"]
        lat = [e["latency_ms"] for e in llm]
        return {
            "llm_calls": len(llm),
            "embed_calls": sum(e["kind"] == "embed" for e in evs),
            "tool_calls": len(tools),
            "tool_errors": sum(not e["ok"] for e in tools),
            "tokens_in": sum(e["tokens_in"] for e in evs),
            "tokens_out": sum(e["tokens_out"] for e in evs),
            "cost_usd": round(sum(e["cost_usd"] for e in evs), 6),
            "llm_latency_p50_ms": float(np.percentile(lat, 50)) if lat else 0.0,
            "llm_latency_p95_ms": float(np.percentile(lat, 95)) if lat else 0.0,
            "errors": sum(not e["ok"] for e in evs),
        }

    def report(self, label: str = "", run_id: str | None = None) -> None:
        s = self.summary(run_id)
        print(f"meter{(' - ' + label) if label else ''}: {s['llm_calls']} LLM calls, "
              f"{s['tool_calls']} tool calls, {s['tokens_in']:,} in / {s['tokens_out']:,} out "
              f"tokens -> ${s['cost_usd']:.4f} | p50 {s['llm_latency_p50_ms']:.0f} ms")

    def flush(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else config.TRACES_DIR / f"trace-{int(time.time())}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        return path

    def reset(self) -> None:
        self.events.clear()


TELEMETRY = Telemetry()
