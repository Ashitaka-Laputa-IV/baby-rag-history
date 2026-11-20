"""r6: RAPTOR. Cluster chunk embeddings recursively and summarize each cluster.

Level 1 clusters leaves; level 2 clusters the level-1 summaries. Retrieval is
collapsed-tree: all summary nodes live in one Chroma collection.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

from babyrag.config import CHROMA_DIR, CHUNKS, RAPTOR_NODES, ensure_dirs, load_settings
from babyrag.llm import LLMClient, build_embeddings
from babyrag.retrieval import read_jsonl

SUMMARY_PROMPT = """把下面这些中文维基片段合并成一段 150–250 字的摘要。
必须保留实体名、地点、年份与关键数字；不要编造片段里没有的信息。

片段：
{text}
"""
_local = threading.local()


def client() -> LLMClient:
    if not hasattr(_local, "client"):
        _local.client = LLMClient(load_settings(), max_output_tokens=400)
    return _local.client


def summarize(text: str) -> tuple[str, int, int, bool]:
    result = client().invoke(SUMMARY_PROMPT.format(text=text), stage="raptor_summary")
    return result.text.strip(), result.tokens_in, result.tokens_out, result.cached


def norm(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def kmeans(X: np.ndarray, k: int, *, seed: int = 42, iters: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(X)))
    C = X[rng.choice(len(X), k, replace=False)].copy()
    Xn = norm(X)
    for _ in range(iters):
        labels = (Xn @ norm(C).T).argmax(axis=1)
        new = np.zeros_like(C)
        counts = np.zeros(k)
        for i, label in enumerate(labels):
            new[label] += X[i]
            counts[label] += 1
        empty = counts == 0
        if empty.any():
            new[empty] = X[rng.choice(len(X), int(empty.sum()), replace=False)]
            counts[empty] = 1
        C = new / counts[:, None]
    return (Xn @ norm(C).T).argmax(axis=1)


def build_level(rows: list[dict], embeddings: list[list[float]], *, level: int,
                per_cluster: int = 9, workers: int = 12) -> list[dict]:
    X = np.asarray(embeddings, dtype=np.float32)
    k = math.ceil(len(rows) / per_cluster)
    labels = kmeans(X, k)
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)
    print(f"level {level}: {len(rows)} nodes -> {len(groups)} clusters")

    nodes: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for label, members in groups.items():
            text = "\n\n".join(rows[i]["text"] for i in members)[:9000]
            futures[pool.submit(summarize, text)] = (label, members)
        for future in as_completed(futures):
            label, members = futures[future]
            summary, tin, tout, cached = future.result()
            nodes.append({
                "node_id": f"raptor:L{level}:{label:04d}",
                "level": level,
                "text": summary,
                "child_ids": [rows[i].get("node_id") or rows[i].get("chunk_id") for i in members],
                "doc_ids": sorted({rows[i].get("doc_id", "") for i in members if rows[i].get("doc_id")}),
                "tokens_in": tin, "tokens_out": tout, "cached": cached,
            })
    return nodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cluster", type=int, default=9)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    ensure_dirs()
    settings = load_settings()
    embeddings_model = build_embeddings(settings)
    chunks = read_jsonl(CHUNKS)
    print(f"leaves: {len(chunks)}")

    leaves = [{"node_id": c["chunk_id"], "doc_id": c["doc_id"],
               "text": c["text"], "level": 0} for c in chunks]
    vectors = embeddings_model.embed_documents([c["text"] for c in chunks])
    print(f"embedded leaves: {len(vectors)}")

    all_nodes: list[dict] = []
    current = leaves
    current_vectors = vectors
    for level in range(1, args.levels + 1):
        nodes = build_level(current, current_vectors, level=level,
                            per_cluster=args.per_cluster, workers=args.workers)
        all_nodes.extend(nodes)
        current = [{"node_id": n["node_id"], "doc_id": "", "text": n["text"],
                    "level": level} for n in nodes]
        current_vectors = embeddings_model.embed_documents([n["text"] for n in nodes])

    with RAPTOR_NODES.open("w", encoding="utf-8", newline="\n") as handle:
        for node in all_nodes:
            handle.write(json.dumps(node, ensure_ascii=False) + "\n")
    print(f"wrote {RAPTOR_NODES} ({len(all_nodes)} summary nodes)")

    store = Chroma(collection_name="raptor", embedding_function=embeddings_model,
                   persist_directory=str(CHROMA_DIR),
                   collection_metadata={"hnsw:space": "cosine"})
    if store._collection.count():
        store.delete_collection()
        store = Chroma(collection_name="raptor", embedding_function=embeddings_model,
                       persist_directory=str(CHROMA_DIR),
                       collection_metadata={"hnsw:space": "cosine"})
    store.add_documents(
        [Document(page_content=n["text"], metadata={
            "chunk_id": n["node_id"], "doc_id": "",
            "doc_ids": "|".join(n.get("doc_ids") or []),
            "title": f"RAPTOR L{n['level']}", "level": n["level"]}) for n in all_nodes],
        ids=[n["node_id"] for n in all_nodes],
    )
    print(f"indexed raptor collection: {store._collection.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
