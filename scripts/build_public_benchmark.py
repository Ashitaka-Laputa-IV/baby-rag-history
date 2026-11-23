"""Build a small RAGBench (emanual) external anchor.

25 English user-manual questions, each with its own supporting documents. We
deduplicate documents, chunk them 512/64, and map RAGBench's sentence-support
keys back to gold doc ids and one evidence sentence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset  # noqa: E402

from babyrag.chunking import DocumentChunker  # noqa: E402
from babyrag.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA, DELIM, ENCODING, INCLUDE_DELIM  # noqa: E402

OUT = DATA / "public"
COUNT = 25


def digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("galileo-ai/ragbench", "emanual", split="test").select(range(COUNT))
    chunker = DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                              encoding=ENCODING, delim=DELIM, include_delim=INCLUDE_DELIM)

    doc_ids: dict[str, str] = {}
    docs: list[dict] = []
    chunks: list[dict] = []
    queries: list[dict] = []

    for row in ds:
        row_doc_ids = []
        sentence_map: dict[str, str] = {}
        key_to_doc: dict[str, str] = {}
        for doc_index, text in enumerate(row["documents"]):
            key = digest(text)
            doc_id = f"pub-{key}"
            row_doc_ids.append(doc_id)
            if doc_id in doc_ids:
                continue
            doc_ids[doc_id] = doc_id
            title = text.strip().splitlines()[0][:80]
            doc = {"doc_id": doc_id, "title": title, "theme": "public",
                   "text": text[:8000], "url": "", "chars": len(text[:8000]),
                   "license": "CC-BY-4.0"}
            docs.append(doc)
            for chunk in chunker.chunk_document(doc):
                chunks.append(chunk)
            for sent_key, sentence in row["documents_sentences"][doc_index]:
                sentence_map[sent_key] = sentence
                key_to_doc[sent_key] = doc_id

        support_keys = []
        for info in row["sentence_support_information"]:
            support_keys.extend(info.get("supporting_sentence_keys") or [])
        gold_docs = sorted({key_to_doc[k] for k in support_keys if k in key_to_doc}) or row_doc_ids
        evidence = sentence_map.get(support_keys[0], "") if support_keys else ""

        queries.append({
            "qid": row["id"], "type": "public", "theme": "public",
            "question": row["question"], "gold_answer": row["response"],
            "gold_doc_ids": gold_docs, "evidence": evidence,
            "should_abstain": False,
        })

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write(OUT / "corpus.jsonl", docs)
    write(OUT / "chunks.jsonl", chunks)
    write(OUT / "queries.jsonl", queries)
    (OUT / "manifest.json").write_text(json.dumps({
        "source": "galileo-ai/ragbench:emanual:test", "license": "CC-BY-4.0",
        "questions": len(queries), "documents": len(docs), "chunks": len(chunks),
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"questions={len(queries)} docs={len(docs)} chunks={len(chunks)}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
