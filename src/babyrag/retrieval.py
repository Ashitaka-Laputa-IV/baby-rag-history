"""Retrieval services: dense, BM25, RRF, rerank, contextual, RAPTOR."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import jieba
import tiktoken
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from .config import (CHROMA_DIR, CHUNKS, CONTEXTUAL, RAPTOR_NODES, Settings,
                     ENCODING)
from .llm import build_embeddings, rerank
from .types import Chunk

RRF_K = 60
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")
_INIT_LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if TOKEN_RE.search(t)]


class DenseRetriever:
    def __init__(self, settings: Settings, collection: str) -> None:
        # Chroma's embedded client initialises shared state; serialise it so
        # four evaluation threads cannot race on the same persist directory.
        with _INIT_LOCK:
            self.store = Chroma(
                collection_name=collection,
                embedding_function=build_embeddings(settings),
                persist_directory=str(CHROMA_DIR),
                collection_metadata={"hnsw:space": "cosine"},
            )
        self.collection = collection

    def search(self, query: str, k: int = 10, *, source: str = "dense") -> list[Chunk]:
        pairs = self.store.similarity_search_with_score(query, k=k)
        out: list[Chunk] = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            meta = doc.metadata or {}
            doc_id = str(meta.get("doc_id", ""))
            raw_ids = str(meta.get("doc_ids", "") or "")
            doc_ids = [x for x in raw_ids.split("|") if x] or ([doc_id] if doc_id else [])
            out.append(Chunk(
                chunk_id=str(meta.get("chunk_id", "")),
                doc_id=doc_id, doc_ids=doc_ids,
                title=str(meta.get("title", "")),
                text=doc.page_content,
                score=float(score),
                rank=rank,
                source=source,
                level=int(meta.get("level", 0) or 0),
            ))
        return out


class BM25Retriever:
    def __init__(self, rows: list[dict], *, text_field: str = "text", source: str = "bm25") -> None:
        self.rows = rows
        self.text_field = text_field
        self.source = source
        self._tokens = [tokenize(r[text_field]) for r in rows]
        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, k: int = 10) -> list[Chunk]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out: list[Chunk] = []
        for rank, i in enumerate(order, start=1):
            row = self.rows[i]
            if scores[i] <= 0:
                continue
            out.append(Chunk(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                doc_ids=row.get("doc_ids") or [row["doc_id"]],
                title=row.get("title", ""),
                text=row[self.text_field], score=float(scores[i]), rank=rank,
                source=self.source, level=int(row.get("level", 0) or 0),
            ))
        return out


class Services:
    """Lazily-built, process-wide retrieval services (never put in state)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.paths = settings.paths
        self._dense: DenseRetriever | None = None
        self._contextual_dense: DenseRetriever | None = None
        self._raptor: DenseRetriever | None = None
        self._bm25: BM25Retriever | None = None
        self._contextual_bm25: BM25Retriever | None = None
        self._chunks: list[dict] | None = None
        self._contextual: dict[str, dict] | None = None
        self._enc = tiktoken.get_encoding(ENCODING)

    @property
    def chunks(self) -> list[dict]:
        if self._chunks is None:
            self._chunks = read_jsonl(self.paths["chunks"])
        return self._chunks

    @property
    def contextual(self) -> dict[str, dict]:
        if self._contextual is None:
            path = self.paths["contextual"]
            self._contextual = ({r["chunk_id"]: r for r in read_jsonl(path)}
                                if path.exists() else {})
        return self._contextual

    @property
    def dense(self) -> DenseRetriever:
        if self._dense is None:
            self._dense = DenseRetriever(self.settings, self.paths["dense"])
        return self._dense

    @property
    def contextual_dense(self) -> DenseRetriever:
        if self._contextual_dense is None:
            self._contextual_dense = DenseRetriever(
                self.settings, self.paths["contextual_dense"])
        return self._contextual_dense

    @property
    def raptor(self) -> DenseRetriever:
        if self._raptor is None:
            self._raptor = DenseRetriever(self.settings, self.paths["raptor_dense"])
        return self._raptor

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.chunks)
        return self._bm25

    @property
    def contextual_bm25(self) -> BM25Retriever:
        if self._contextual_bm25 is None:
            rows = [{"chunk_id": k, "doc_id": v["doc_id"], "title": v.get("title", ""),
                     "text": v["contextual_text"], "level": 0}
                    for k, v in self.contextual.items()]
            self._contextual_bm25 = BM25Retriever(rows, source="bm25_contextual")
        return self._contextual_bm25

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))


SERVICES: Services | None = None


def get_services(settings: Settings) -> Services:
    global SERVICES
    if SERVICES is None:
        SERVICES = Services(settings)
    return SERVICES


def fuse_rrf(*lists: list[Chunk], k: int = RRF_K, top_n: int = 20) -> list[Chunk]:
    scores: dict[str, float] = {}
    best: dict[str, Chunk] = {}
    for hits in lists:
        for rank, hit in enumerate(hits, start=1):
            cid = hit["chunk_id"]
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best or hit.get("score", 0) > best[cid].get("score", 0):
                best[cid] = hit
    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_n]
    out: list[Chunk] = []
    for rank, cid in enumerate(ordered, start=1):
        hit = dict(best[cid])
        hit["rank"] = rank
        hit["score"] = scores[cid]
        hit["source"] = "rrf"
        out.append(hit)  # type: ignore[arg-type]
    return out


def rerank_hits(settings: Settings, query: str, hits: list[Chunk],
                *, top_n: int = 5) -> list[Chunk]:
    if not hits:
        return []
    pairs = rerank(settings, query, [h["text"] for h in hits], top_n=top_n)
    out: list[Chunk] = []
    for rank, (index, score) in enumerate(pairs, start=1):
        if not (0 <= index < len(hits)):
            continue
        hit = dict(hits[index])
        hit["score"] = float(score)
        hit["rank"] = rank
        hit["source"] = "rerank"
        out.append(hit)  # type: ignore[arg-type]
    return out


def build_context(hits: list[Chunk], *, services: Services,
                  budget: int = 4000) -> tuple[str, list[Chunk]]:
    kept: list[Chunk] = []
    pieces: list[str] = []
    used = 0
    for hit in hits:
        title = hit.get("title", "")
        body = hit["text"]
        piece = f"[{len(kept) + 1}] （{title}）{body}" if title else f"[{len(kept) + 1}] {body}"
        cost = services.count_tokens(piece)
        if used + cost > budget:
            continue
        kept.append(hit)
        pieces.append(piece)
        used += cost
    return "\n\n".join(pieces), kept


def extract_citations(answer: str, hits: list[Chunk]) -> list[str]:
    cited: list[str] = []
    for match in re.finditer(r"\[(\d{1,2})\]", answer):
        index = int(match.group(1)) - 1
        if 0 <= index < len(hits):
            cid = hits[index]["chunk_id"]
            if cid and cid not in cited:
                cited.append(cid)
    return cited
