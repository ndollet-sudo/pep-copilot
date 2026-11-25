"""Tools, schemas and the registry - schema-first, validated before execution.

| Tool          | Blast radius | Role                                                   |
|---------------|--------------|--------------------------------------------------------|
| search_peps   | read-only    | hybrid RAG search over the 500 PEPs (chunks + cites)   |
| get_pep       | read-only    | exact metadata of one PEP (status, version, authors)   |
| find_peps     | read-only    | structured filter over metadata (counts, lists)        |
| calculator    | compute      | exact arithmetic via a whitelisted AST (no eval)       |
| final_answer  | control flow | termination: answer + the PEP numbers it relies on     |

Contract: ``dispatch`` NEVER raises. Every problem becomes an observation the model can
read and recover from (unknown tool, invalid arguments, tool failure, empty result).
"""
from __future__ import annotations

import ast
import json
import operator as op
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..corpus import docs_by_pep

# ----------------------------------------------------------------------------- results


@dataclass
class ToolResult:
    ok: bool
    observation: str
    error_type: str | None = None          # unknown_tool | invalid_args | tool_error | empty
    peps: tuple[int, ...] = ()             # PEP numbers surfaced by this call (grounding)


# -------------------------------------------------------------------------- validation
_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "array": list, "object": dict}


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


# ------------------------------------------------------------------------------- tools
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_UN = {ast.UAdd: op.pos, ast.USub: op.neg}


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


def _version_key(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])


def make_tools(retriever, rerank: bool = False) -> dict[str, tuple[dict, Callable[..., ToolResult]]]:
    """Build the registry: name -> (openai tool schema, function)."""
    peps = docs_by_pep()

    def search_peps(query: str, k: int = 5) -> ToolResult:
        ids = retriever.search(query, mode="hybrid", k=k, rerank=rerank)
        if not ids:
            return ToolResult(True, "No passage matches this query. Try different keywords "
                                    "(feature names, module names, PEP titles).", "empty")
        blocks, seen = [], []
        for cid in ids:
            c = retriever.by_id[cid]
            sec = f" > {c['section']}" if c["section"] else ""
            blocks.append(f"[PEP {c['pep']}: {c['title']} ({c['status']}){sec}]\n{c['text'][:900]}")
            seen.append(c["pep"])
        return ToolResult(True, "\n\n".join(blocks), peps=tuple(seen))

    def get_pep(number: int) -> ToolResult:
        d = peps.get(number)
        if d is None:
            return ToolResult(True, f"PEP {number} is not in the knowledge base (500 PEPs). "
                                    "Use search_peps or find_peps to locate the right PEP.",
                              "empty")
        meta = {k: d[k] for k in ("pep", "title", "authors", "status", "type", "created",
                                  "python_version", "replaces", "superseded_by", "url") if d[k]}
        abstract = " ".join(d["text"].split()[:120])
        return ToolResult(True, json.dumps(meta, ensure_ascii=False) + f"\nStart of text: {abstract}...",
                          peps=(number,))

    def find_peps(status: str | None = None, type: str | None = None,  # noqa: A002
                  python_version: str | None = None, title_contains: str | None = None,
                  limit: int = 20) -> ToolResult:
        rows = list(peps.values())
        if status:
            rows = [d for d in rows if d["status"].lower() == status.lower()]
        if type:
            rows = [d for d in rows if d["type"].lower() == type.lower()]
        if python_version:
            rows = [d for d in rows if d["python_version"] and
                    _version_key(d["python_version"]) == _version_key(python_version)]
        if title_contains:
            rows = [d for d in rows if title_contains.lower() in d["title"].lower()]
        rows.sort(key=lambda d: d["pep"])
        if not rows:
            return ToolResult(True, "count: 0 - no PEP matches these filters.", "empty")
        listing = "\n".join(f"PEP {d['pep']}: {d['title']} ({d['status']}, {d['type']}"
                            f"{', Python ' + d['python_version'] if d['python_version'] else ''})"
                            for d in rows[:limit])
        more = f"\n... and {len(rows) - limit} more" if len(rows) > limit else ""
        return ToolResult(True, f"count: {len(rows)}\n{listing}{more}",
                          peps=tuple(d["pep"] for d in rows[:limit]))

    def final_answer(answer: str, sources: list[int] | None = None) -> ToolResult:
        return ToolResult(True, f"Final answer recorded: {answer}")

    statuses = sorted({d["status"] for d in peps.values()})
    types = sorted({d["type"] for d in peps.values()})
    schemas = {
        "search_peps": {
            "description": "Semantic + keyword search over the full text of 500 Python "
                           "Enhancement Proposals. Use it for any question about what a PEP "
                           "says, why a feature was designed a certain way, syntax, rationale. "
                           "Returns the most relevant passages with their PEP number.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search query, in English works best."},
                "k": {"type": "integer", "minimum": 1, "maximum": 10,
                      "description": "Number of passages (default 5)."}},
                "required": ["query"]}},
        "get_pep": {
            "description": "Exact metadata of one PEP by number: title, authors, status, type, "
                           "creation date, target Python version, replaces/superseded-by.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "minimum": 1, "description": "PEP number, e.g. 572"}},
                "required": ["number"]}},
        "find_peps": {
            "description": "Filter PEPs by metadata and COUNT them. Use for questions like "
                           "'how many Final PEPs target Python 3.11' or 'list Process PEPs'.",
            "parameters": {"type": "object", "properties": {
                "status": {"type": "string", "enum": statuses},
                "type": {"type": "string", "enum": types},
                "python_version": {"type": "string", "description": "e.g. '3.11'"},
                "title_contains": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                "required": []}},
        "calculator": {
            "description": "Evaluate an arithmetic expression exactly. Use ** for powers. "
                           "Supports + - * / // % ** and parentheses.",
            "parameters": {"type": "object", "properties": {
                "expression": {"type": "string", "description": "e.g. (2023 - 2014) * 12"}},
                "required": ["expression"]}},
        "final_answer": {
            "description": "Give the final answer to the user. Call exactly once, when done. "
                           "List in 'sources' the PEP numbers that support the answer.",
            "parameters": {"type": "object", "properties": {
                "answer": {"type": "string", "description": "The final answer, concise."},
                "sources": {"type": "array", "items": {"type": "integer"},
                            "description": "PEP numbers supporting the answer."}},
                "required": ["answer"]}},
    }
    fns = {"search_peps": search_peps, "get_pep": get_pep, "find_peps": find_peps,
           "calculator": calculator, "final_answer": final_answer}
    return {name: ({"type": "function", "function": {"name": name, **schemas[name]}}, fns[name])
            for name in schemas}


class ToolRegistry:
    """name -> (schema, fn). A call outside the registry cannot execute."""

    def __init__(self, tools: dict[str, tuple[dict, Callable[..., ToolResult]]]):
        self.tools = tools

    @property
    def schemas(self) -> list[dict]:
        return [s for s, _ in self.tools.values()]

    def params(self, name: str) -> dict:
        return self.tools[name][0]["function"]["parameters"]

    def classify(self, name: str, arguments_json: str) -> str:
        if name not in self.tools:
            return "unknown_tool"
        ok, _, _ = validate_args(self.params(name), arguments_json)
        return "ok" if ok else "invalid_args"

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
