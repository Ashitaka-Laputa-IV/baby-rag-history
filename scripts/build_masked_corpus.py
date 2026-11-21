"""Build the masked (hard) corpus: hide titles and add near-duplicate negatives.

- Title strings are replaced with 「（标题已隐藏）」 in both docs and chunks, so
  the 23/30 questions that literally contain the gold title can no longer be
  answered by lexical matching.
- 200 additional docs are mined from the rule-classified pool by sharing a
  2-character title prefix with an existing doc (same surname / place type),
  then chunked with the same 512/64 chunker and appended as `neg-*` docs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from babyrag.chunking import DocumentChunker
from babyrag.config import (CHUNK_OVERLAP, CHUNK_SIZE, DATA, DELIM, ENCODING,
                            INCLUDE_DELIM, variant_paths)

MASK = "（标题已隐藏）"


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def mask_text(text: str, title: str) -> str:
    if title and title in text:
        text = text.replace(title, MASK)
    # also mask the title with spaces/punctuation removed
    compact = "".join(title.split())
    if compact and compact != title and compact in text:
        text = text.replace(compact, MASK)
    return text


def digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", type=int, default=200)
    ap.add_argument("--extra", type=Path, default=DATA / "corpus.rule.jsonl")
    args = ap.parse_args()

    base_docs = read(DATA / "corpus.jsonl")
    base_chunks = read(DATA / "chunks.jsonl")
    titles = {norm(d["title"]): d["title"] for d in base_docs}
    prefixes = {t[:2] for t in titles if len(t) >= 2}
    existing = {d["doc_id"] for d in base_docs}

    negatives: list[dict] = []
    if args.extra.exists():
        pool = [d for d in read(args.extra) if d["doc_id"] not in existing]
        scored = []
        for d in pool:
            key = norm(d["title"])
            shared = key[:2] in prefixes if len(key) >= 2 else False
            scored.append((0 if shared else 1, digest(d["title"]), d))
        scored.sort(key=lambda x: (x[0], x[1]))
        for _, _, doc in scored:
            if len(negatives) >= args.negatives:
                break
            if norm(doc["title"]) in titles:
                continue
            negatives.append({**doc, "doc_id": f"neg-{digest(doc['title'])}",
                              "theme": doc.get("theme", "other")})

    chunker = DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                              encoding=ENCODING, delim=DELIM,
                              include_delim=INCLUDE_DELIM)

    docs: list[dict] = []
    chunks: list[dict] = []
    for doc in base_docs:
        masked = dict(doc)
        masked["text"] = mask_text(doc["text"], doc["title"])
        docs.append(masked)
    for chunk in base_chunks:
        masked = dict(chunk)
        masked["text"] = mask_text(chunk["text"], chunk["title"])
        chunks.append(masked)

    neg_chunks = 0
    for doc in negatives:
        masked = dict(doc)
        masked["text"] = mask_text(doc["text"], doc["title"])
        docs.append(masked)
        for row in chunker.chunk_document(masked):
            row["text"] = mask_text(row["text"], masked["title"])
            chunks.append(row)
            neg_chunks += 1

    paths = variant_paths("masked")
    paths["corpus"].write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n",
        encoding="utf-8")
    paths["chunks"].write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8")

    print(f"docs={len(docs)} (base={len(base_docs)} negatives={len(negatives)})")
    print(f"chunks={len(chunks)} (base={len(base_chunks)} negatives={neg_chunks})")
    print("themes=" + " ".join(f"{k}:{v}" for k, v in
                               Counter(d["theme"] for d in docs).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
