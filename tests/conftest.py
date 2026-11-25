"""Shared test doubles: a scripted fake LLM and a stub retriever (no network, no API key)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def tool_call(name: str, args: dict | str, cid: str = "c1"):
    arguments = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=cid, type="function",
                           function=SimpleNamespace(name=name, arguments=arguments))


def response(tool_calls=None, content: str | None = None, tin: int = 100, tout: int = 20):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout))


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


class StubRetriever:
    def __init__(self):
        self.by_id = {
            "pep-0572#c0": {"chunk_id": "pep-0572#c0", "doc_id": "pep-0572", "pep": 572,
                            "title": "Assignment Expressions", "status": "Final",
                            "section": "Abstract", "text": "NAME := expr, the walrus operator."},
        }

    def search(self, query, mode="hybrid", k=5, rerank=False):
        return [] if "nothing" in query else list(self.by_id)[:k]


@pytest.fixture
def registry():
    from pepcopilot.agent.tools import ToolRegistry, make_tools
    return ToolRegistry(make_tools(StubRetriever()))
