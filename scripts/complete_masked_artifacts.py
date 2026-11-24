"""Complete the masked artifacts without paying for a second contextual pass.

The original contextual prefixes were generated with titles in the prompt, so
they can leak the title.  Here we keep the prefix but replace the literal title
with 「（标题已隐藏）」, then re-embed.  Embeddings are free, so this is ~1 minute
instead of ~45 minutes and ~$1.4 of LLM calls.  RAPTOR summaries get the same
treatment.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from babyrag.config import (CHROMA_DIR, DATA, ensure_dirs, load_settings,
                            variant_paths)
from babyrag.llm import build_embeddings

MASK = "（标题已隐藏）"


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_title(text: str, titles: list[str]) -> str:
    for title in titles:
        if title:
            text = text.replace(title, MASK)
    return text


def main() -> int:
    ensure_dirs()
    settings = load_settings()
    paths = variant_paths("masked")
    masked_chunks = read(paths["chunks"])
    masked_docs = {d["doc_id"]: d for d in read(paths["corpus"])}
    original_contextual = {r["chunk_id"]: r for r in read(DATA / "contextual.jsonl")}
    partial = {r["chunk_id"]: r for r in read(paths["contextual"])}
    titles = {doc_id: d["title"] for doc_id, d in masked_docs.items()}

    rows: list[dict] = []
    for chunk in masked_chunks:
        cid, doc_id = chunk["chunk_id"], chunk["doc_id"]
        if cid in partial and partial[cid].get("prefix"):
            rows.append(partial[cid])
            continue
        title = titles.get(doc_id, "")
        if cid in original_contextual:
            prefix = strip_title(original_contextual[cid].get("prefix", ""), [title])
        else:
            prefix = ""
        rows.append({
            "chunk_id": cid, "doc_id": doc_id, "title": MASK if title else "",
            "prefix": prefix,
            "contextual_text": f"{prefix}\n{chunk['text']}" if prefix else chunk["text"],
        })
    write(paths["contextual"], rows)
    print(f"contextual.masked rows: {len(rows)} "
          f"(generated={len(partial)}, reused={len(rows) - len(partial)})")

    raptor_src = read(DATA / "raptor.jsonl")
    raptor_rows: list[dict] = []
    for node in raptor_src:
        doc_titles = [titles.get(d, "") for d in node.get("doc_ids", [])]
        raptor_rows.append({**node, "text": strip_title(node["text"], doc_titles)})
    write(paths["raptor"], raptor_rows)
    print(f"raptor.masked nodes: {len(raptor_rows)}")

    embeddings = build_embeddings(settings)
    for key, collection, docs in (
        ("contextual", paths["contextual_dense"],
         [(r["chunk_id"], r["contextual_text"], r["doc_id"], r["title"], 0) for r in rows]),
        ("raptor", paths["raptor_dense"],
         [(r["node_id"], r["text"], "", f"RAPTOR L{r['level']}", r["level"]) for r in raptor_rows]),
    ):
        store = Chroma(collection_name=collection, embedding_function=embeddings,
                       persist_directory=str(CHROMA_DIR),
                       collection_metadata={"hnsw:space": "cosine"})
        if store._collection.count():
            store.delete_collection()
            store = Chroma(collection_name=collection, embedding_function=embeddings,
                           persist_directory=str(CHROMA_DIR),
                           collection_metadata={"hnsw:space": "cosine"})
        for start in range(0, len(docs), 128):
            batch = docs[start:start + 128]
            store.add_documents(
                [Document(page_content=text, metadata={
                    "chunk_id": cid, "doc_id": doc_id, "title": title, "level": level,
                    "doc_ids": doc_id}) for cid, text, doc_id, title, level in batch],
                ids=[cid for cid, *_ in batch],
            )
            print(f"  {collection} {min(start + 128, len(docs))}/{len(docs)}", end="\r", flush=True)
        print(f"\nindexed {collection}: {store._collection.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
