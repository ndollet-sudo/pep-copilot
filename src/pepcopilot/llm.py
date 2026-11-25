"""LLM access layer: one OpenAI-SDK client, pointed at OpenAI or OpenRouter (config.PROVIDER).

- ``chat()``  : metered chat completion with exponential backoff on transient errors
- ``chat_json()``: chat + JSON parsing + Pydantic validation, with one repair retry
- ``embed()`` : metered, batched embeddings

Transient errors (429, 5xx, timeouts) are retried; permanent ones (bad key,
bad request) fail fast. Every call is recorded by the telemetry layer.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import config
from .telemetry import TELEMETRY

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the provider fails permanently or after all retries."""


def _status(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


class LLMClient:
    def __init__(self, chat_model: str = config.CHAT_MODEL,
                 embed_model: str = config.EMBED_MODEL, api_key: str | None = None):
        from openai import OpenAI  # imported lazily: offline mode never needs it

        key = api_key or os.environ.get(config.API_KEY_ENV)
        if not key:
            raise LLMError(f"Set {config.API_KEY_ENV} (see .env.example) to use the LLM.")
        self.client = OpenAI(api_key=key, base_url=config.BASE_URL, max_retries=0,
                             timeout=60.0)
        self.chat_model, self.embed_model = chat_model, embed_model

    # ------------------------------------------------------------------ retry
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

    # ------------------------------------------------------------------- chat
    def chat(self, messages: list, tools: list | None = None, temperature: float = 0.0,
             json_mode: bool = False, name: str = "chat") -> Any:
        kwargs: dict[str, Any] = dict(model=self.chat_model, messages=messages,
                                      temperature=temperature,
                                      extra_body=config.EXTRA_BODY)
        if tools:
            kwargs["tools"], kwargs["tool_choice"] = tools, "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with TELEMETRY.span("llm", name, model=self.chat_model) as ev:
            resp = self._with_retry(lambda: self.client.chat.completions.create(**kwargs), name)
            if resp.usage is not None:
                ev["tokens_in"] = resp.usage.prompt_tokens or 0
                ev["tokens_out"] = resp.usage.completion_tokens or 0
        return resp

    def chat_json(self, messages: list, schema: Type[T], name: str = "chat_json") -> T:
        """Structured output: JSON mode + Pydantic validation. On a schema violation the
        validation error is sent back to the model once (repair loop), then we give up."""
        msgs = list(messages)
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] +
                   "\n\nReply with a single JSON object matching this JSON schema:\n" +
                   json.dumps(schema.model_json_schema())}
        last_err = ""
        for _ in range(2):
            resp = self.chat(msgs, json_mode=True, name=name)
            raw = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:500]
                msgs = msgs + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": f"Invalid JSON for the schema: {last_err}. "
                                                           "Return only the corrected JSON object."}]
        raise LLMError(f"{name}: model did not return valid JSON ({last_err})")

    # ------------------------------------------------------------------ embed
    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with TELEMETRY.span("embed", "embed", model=self.embed_model, n=len(batch)) as ev:
                resp = self._with_retry(
                    lambda: self.client.embeddings.create(model=self.embed_model, input=batch),
                    "embed")
                u = resp.usage
                ev["tokens_in"] = (getattr(u, "prompt_tokens", None) or
                                   getattr(u, "total_tokens", 0) or 0) if u else 0
            out.extend(d.embedding for d in resp.data)
        return out


def _extract_json(raw: str) -> str:
    """Tolerate ```json fences or prose around the object."""
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return m.group(0)


_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
