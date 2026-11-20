"""Build the base dense index (bge-m3 -> Chroma). BM25 is built in memory."""

from __future__ import annotations

import argparse
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from babyrag.config import CHROMA_DIR, CHUNKS, ensure_dirs, load_settings
from babyrag.llm import build_embeddings
from babyrag.retrieval import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(CHUNKS))
    ap.add_argument("--collection", default="chunks")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    settings = load_settings()
    rows = read_jsonl(__import__("pathlib").Path(args.chunks))
    print(f"chunks: {len(rows)}")
    embeddings = build_embeddings(settings)
    store = Chroma(collection_name=args.collection, embedding_function=embeddings,
                   persist_directory=str(CHROMA_DIR),
                   collection_metadata={"hnsw:space": "cosine"})
    if args.rebuild:
        store.delete_collection()
        store = Chroma(collection_name=args.collection, embedding_function=embeddings,
                       persist_directory=str(CHROMA_DIR),
                       collection_metadata={"hnsw:space": "cosine"})
    existing = store._collection.count()
    if existing == len(rows) and not args.rebuild:
        print(f"already indexed ({existing})")
        return 0
    if existing:
        print(f"collection has {existing}, expected {len(rows)}; use --rebuild")
        return 1

    started = time.perf_counter()
    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        docs = [Document(page_content=r["text"], metadata={
            "chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
            "title": r.get("title", ""), "theme": r.get("theme", ""),
            "level": 0,
        }) for r in batch]
        store.add_documents(docs, ids=[r["chunk_id"] for r in batch])
        print(f"  {min(start + args.batch, len(rows))}/{len(rows)}", end="\r", flush=True)
    print(f"\nindexed {store._collection.count()} vectors in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
