"""Model factories and the cost-aware LLM client.

OpenCode Go requires `x-opencode-session` on every request; without it the
gateway answers HTTP 400.  Embeddings and reranking go to SiliconFlow, where
bge-m3 and bge-reranker-v2-m3 are free for this account.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .cache import Cache
from .config import (CACHE_DB, RESULTS, EMBED_MODEL, RERANK_MODEL, Settings,
                     PRICE_INPUT, PRICE_OUTPUT, PRICE_CACHE_READ)
from .types import NodeRecord

COST_LOG = RESULTS / "cost.jsonl"


def _key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@dataclass
class LLMResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    cached: bool = False
    latency_ms: int = 0


class LLMClient:
    """Chat wrapper that caches by content and records real token usage."""

    def __init__(self, settings: Settings, *, max_output_tokens: int = 1024,
                 temperature: float = 0.0) -> None:
        self.settings = settings
        self.model = settings.model
        self.cache = Cache(CACHE_DB)
        self._max_output = max_output_tokens
        self._temperature = temperature
        self._llm = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=temperature,
            max_tokens=max_output_tokens,
            timeout=120,
            max_retries=2,
            default_headers={"x-opencode-session": settings.session_id},
        )
        self.spent_usd = 0.0

    def invoke(self, prompt: str, *, stage: str, system: str | None = None,
               prompt_version: str = "v1", use_cache: bool = True,
               extra_key: str = "") -> LLMResult:
        key = _key(stage, prompt_version, self.model, system or "", prompt, extra_key)
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                try:
                    payload = json.loads(hit)
                except json.JSONDecodeError:
                    payload = {"text": hit}
                return LLMResult(
                    text=payload.get("text", ""),
                    tokens_in=int(payload.get("tokens_in", 0)),
                    tokens_out=int(payload.get("tokens_out", 0)),
                    cached=True,
                )
        started = time.perf_counter()
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        response = self._invoke_with_backoff(messages)
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = _message_text(response)
        tin, tout, cached_in = _usage(response, messages, text)
        cost = (tin * PRICE_INPUT + tout * PRICE_OUTPUT + cached_in * PRICE_CACHE_READ)
        self.spent_usd += cost
        if use_cache:
            self.cache.put(key, json.dumps(
                {"text": text, "tokens_in": tin, "tokens_out": tout},
                ensure_ascii=False), self.model)
        record = {
            "ts": time.time(), "stage": stage, "model": self.model,
            "tokens_in": tin, "tokens_out": tout, "cached_in": cached_in,
            "usd": round(cost, 8), "latency_ms": latency_ms,
            "cache_key": key,
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        with COST_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return LLMResult(text=text, tokens_in=tin, tokens_out=tout,
                         usd=cost, latency_ms=latency_ms)

    def _invoke_with_backoff(self, messages: list[Any]):
        """Retry 429/5xx/timeout with jittered exponential backoff."""
        import random

        last: Exception | None = None
        for attempt in range(5):
            try:
                return self._llm.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - classify by text, gateway-specific
                text = f"{type(exc).__name__}: {exc}"
                retryable = any(s in text for s in ("429", "Rate limit", "rate_limit",
                                                    "500", "502", "503", "504", "timeout",
                                                    "Timeout", "overloaded",
                                                    "Connection error", "Connection"))
                last = exc
                if not retryable or attempt == 4:
                    raise
                time.sleep(min(30.0, 2.0 * (2 ** attempt)) + random.random())
        raise last  # pragma: no cover

    def record(self, result: LLMResult, node: str, note: str = "") -> NodeRecord:
        return NodeRecord(
            node=node, tokens_in=result.tokens_in, tokens_out=result.tokens_out,
            llm_calls=0 if result.cached else 1, cache_hits=1 if result.cached else 0,
            latency_ms=result.latency_ms, usd=result.usd, note=note,
        )

    def budget_left(self) -> float:
        return self.settings.max_usd - self.spent_usd


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _usage(response: Any, messages: list[Any], text: str) -> tuple[int, int, int]:
    """Prefer provider usage; fall back to tiktoken estimates."""
    meta = getattr(response, "usage_metadata", None) or {}
    tin = int(meta.get("input_tokens", 0) or 0)
    tout = int(meta.get("output_tokens", 0) or 0)
    cached = 0
    details = meta.get("input_token_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cache_read", 0) or 0)
    if tin and tout:
        return tin, tout, cached
    raw = getattr(response, "response_metadata", None) or {}
    usage = raw.get("token_usage") or raw.get("usage") or {}
    if isinstance(usage, dict):
        tin = tin or int(usage.get("prompt_tokens", 0) or 0)
        tout = tout or int(usage.get("completion_tokens", 0) or 0)
        cached = cached or int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
    if tin or tout:
        return tin, tout, cached
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tin = sum(len(enc.encode(m.content)) for m in messages)
    tout = len(enc.encode(text))
    return tin, tout, 0


def build_embeddings(settings: Settings, *, batch_size: int = 32) -> OpenAIEmbeddings:
    if not settings.silicon_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not set")
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        base_url=settings.silicon_url,
        api_key=settings.silicon_key,
        check_embedding_ctx_length=False,
        chunk_size=batch_size,
    )


def rerank(settings: Settings, query: str, documents: list[str],
           *, top_n: int = 5, timeout: float = 60.0) -> list[tuple[int, float]]:
    """Return (index, relevance_score) pairs from SiliconFlow's reranker."""
    if not settings.silicon_key or not documents:
        return list(enumerate([0.0] * len(documents)))[:top_n]
    response = httpx.post(
        f"{settings.silicon_url}/rerank",
        headers={"Authorization": f"Bearer {settings.silicon_key}"},
        json={"model": RERANK_MODEL, "query": query,
              "documents": documents, "top_n": min(top_n, len(documents))},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return [(int(r["index"]), float(r["relevance_score"])) for r in body["results"]]
