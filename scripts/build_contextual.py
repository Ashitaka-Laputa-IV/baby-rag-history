"""r5: Contextual Retrieval. Generate one locator prefix per chunk, then index.

Resumable: existing `data/contextual.jsonl` rows are reused, so a rerun never
pays twice for the same chunk.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from babyrag.config import CHROMA_DIR, ensure_dirs, load_settings, variant_paths
from babyrag.llm import LLMClient, build_embeddings
from babyrag.prompts import CONTEXTUAL as CONTEXT_PROMPT
from babyrag.retrieval import read_jsonl

_local = threading.local()


def client() -> LLMClient:
    if not hasattr(_local, "client"):
        _local.client = LLMClient(load_settings(), max_output_tokens=200)
    return _local.client


MASK = "（标题已隐藏）"


def make_prefix(row: dict, doc_text: str, *, masked: bool = False) -> dict:
    prompt = CONTEXT_PROMPT.format(title=MASK if masked else row.get("title", ""),
                                   doc=doc_text[:1800], chunk=row["text"])
    result = client().invoke(prompt, stage=f"contextual{'-masked' if masked else ''}",
                             prompt_version="v1", system="你是中文维基百科的编辑。")
    prefix = result.text.strip().replace("\n", " ")
    return {
        "chunk_id": row["chunk_id"], "doc_id": row["doc_id"],
        "title": row.get("title", ""), "prefix": prefix,
        "contextual_text": f"{prefix}\n{row['text']}",
        "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
        "cached": result.cached,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variant", default="original", choices=["original", "masked"])
    ap.add_argument("--index", action="store_true", help="also build the Chroma collection")
    args = ap.parse_args()

    ensure_dirs()
    settings = load_settings()
    paths = variant_paths(args.variant)
    masked = args.variant == "masked"
    rows = read_jsonl(paths["chunks"])
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(paths["corpus"])}
    if args.limit:
        rows = rows[: args.limit]

    existing: dict[str, dict] = {}
    if paths["contextual"].exists():
        existing = {r["chunk_id"]: r for r in read_jsonl(paths["contextual"])}
    todo = [r for r in rows if r["chunk_id"] not in existing]
    print(f"chunks={len(rows)} cached={len(existing)} todo={len(todo)}")

    paths["contextual"].parent.mkdir(parents=True, exist_ok=True)
    out = paths["contextual"].open("a", encoding="utf-8")
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(make_prefix, r, docs.get(r["doc_id"], ""),
                                   masked=masked): r for r in todo}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:  # one bad chunk must not stop the build
                    print(f"FAILED {futures[future]['chunk_id']}: {exc}")
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                existing[row["chunk_id"]] = row
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(todo)}", end="\r", flush=True)
    finally:
        out.close()
    print(f"\ncontextual rows: {len(existing)}")

    if args.index:
        embeddings = build_embeddings(settings)
        collection = paths["contextual_dense"]
        store = Chroma(collection_name=collection, embedding_function=embeddings,
                       persist_directory=str(CHROMA_DIR),
                       collection_metadata={"hnsw:space": "cosine"})
        if store._collection.count():
            store.delete_collection()
            store = Chroma(collection_name=collection, embedding_function=embeddings,
                           persist_directory=str(CHROMA_DIR),
                           collection_metadata={"hnsw:space": "cosine"})
        items = list(existing.values())
        for start in range(0, len(items), 128):
            batch = items[start:start + 128]
            store.add_documents(
                [Document(page_content=r["contextual_text"], metadata={
                    "chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
                    "title": r.get("title", ""), "level": 0}) for r in batch],
                ids=[r["chunk_id"] for r in batch],
            )
            print(f"  index {min(start + 128, len(items))}/{len(items)}", end="\r", flush=True)
        print(f"\nindexed contextual collection: {store._collection.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
