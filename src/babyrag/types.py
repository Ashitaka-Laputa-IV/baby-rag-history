"""Serializable contracts shared by nodes and the evaluator."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class Chunk(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    doc_ids: list[str]
    title: str
    text: str
    score: float
    rank: int
    source: str          # dense | bm25 | hyde | multiquery | raptor | rerank | agentic
    level: int           # RAPTOR level; leaf = 0
    context: str         # contextual prefix, if any


class NodeRecord(TypedDict, total=False):
    node: str
    tokens_in: int
    tokens_out: int
    llm_calls: int
    cache_hits: int
    latency_ms: int
    usd: float
    note: str


class Cost(TypedDict, total=False):
    tokens_in: int
    tokens_out: int
    llm_calls: int
    cache_hits: int
    latency_ms: int
    usd: float


class Step(TypedDict, total=False):
    node: str
    detail: str


class RAGState(TypedDict, total=False):
    # input / plan
    qid: str
    question: str
    mode: Literal["ladder", "adaptive"]
    rung: str
    plan: list[str]
    cursor: int
    route_reason: str
    # query
    qtype: str
    query_variants: list[str]
    # retrieval
    dense_hits: list[Chunk]
    bm25_hits: list[Chunk]
    multi_hits: list[list[Chunk]]
    hits: list[Chunk]
    # generation
    answer: str
    citations: list[str]
    # control
    grade: Literal["correct", "ambiguous", "incorrect"]
    retries: int
    abstain: bool
    # observability
    node_records: Annotated[list[NodeRecord], operator.add]
    trajectory: Annotated[list[Step], operator.add]
    cost: Cost
    error: str | None


class AnswerResult(TypedDict, total=False):
    qid: str
    rung: str
    question: str
    answer: str
    abstain: bool
    citations: list[str]
    hits: list[Chunk]
    plan: list[str]
    cost: Cost
    node_records: list[NodeRecord]
    error: str


def add_cost(records: list[NodeRecord]) -> Cost:
    out: Cost = {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0,
                 "cache_hits": 0, "latency_ms": 0, "usd": 0.0}
    for r in records:
        out["tokens_in"] += int(r.get("tokens_in", 0))
        out["tokens_out"] += int(r.get("tokens_out", 0))
        out["llm_calls"] += int(r.get("llm_calls", 0))
        out["cache_hits"] += int(r.get("cache_hits", 0))
        out["latency_ms"] += int(r.get("latency_ms", 0))
        out["usd"] += float(r.get("usd", 0.0))
    return out


Json = dict[str, Any]
