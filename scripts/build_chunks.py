"""Build data/chunks.jsonl from data/corpus.jsonl.

    python -m scripts.build_chunks

Self-checks run before anything is written; any failure aborts with exit code 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from babyrag.chunking import DocumentChunker

from . import params


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chunk_corpus(docs: list[dict], chunker: DocumentChunker) -> list[dict]:
    rows: list[dict] = []
    for doc in docs:
        rows.extend(chunker.chunk_document(doc))
    return rows


def find_problems(rows: list[dict], docs: list[dict]) -> dict[str, list[str]]:
    """Return {problem: examples}; empty dict means the output is trustworthy."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[row["doc_id"]].append(row)
    text_by_doc = {doc["doc_id"]: doc["text"] for doc in docs}

    problems: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        chunks = sorted(by_doc.get(doc["doc_id"], []), key=lambda r: r["seq"])
        if not chunks:
            problems["documents without chunks"].append(doc["doc_id"])
            continue
        if chunks[0]["start"] != 0:
            problems["chunks not starting at 0"].append(chunks[0]["chunk_id"])
        if chunks[-1]["end"] != len(doc["text"]):
            problems["chunks not reaching the end"].append(doc["doc_id"])
        for position, chunk in enumerate(chunks):
            if chunk["seq"] != position:
                problems["non-contiguous seq"].append(chunk["chunk_id"])
            if doc["text"][chunk["start"]:chunk["end"]] != chunk["text"]:
                problems["offset mismatches"].append(chunk["chunk_id"])
            if chunk["tokens"] > params.CHUNK_SIZE:
                problems["oversize chunks"].append(chunk["chunk_id"])
        for previous, following in zip(chunks, chunks[1:]):
            if following["start"] > previous["end"] or following["start"] < previous["start"]:
                problems["gaps or out-of-order chunks"].append(following["chunk_id"])

    duplicated_chunks = [cid for cid, n in Counter(r["chunk_id"] for r in rows).items() if n > 1]
    if duplicated_chunks:
        problems["duplicate chunk ids"] = duplicated_chunks
    duplicated_docs = [did for did, n in Counter(d["doc_id"] for d in docs).items() if n > 1]
    if duplicated_docs:
        problems["duplicate doc ids"] = duplicated_docs
    return dict(problems)


def overlap_stats(rows: list[dict], docs: list[dict], encoding: str) -> tuple[int, int, int]:
    import tiktoken

    encoder = tiktoken.get_encoding(encoding)
    text_by_doc = {doc["doc_id"]: doc["text"] for doc in docs}
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[row["doc_id"]].append(row)

    pairs = with_overlap = 0
    overlap_tokens: list[int] = []
    for doc_id, chunks in by_doc.items():
        chunks.sort(key=lambda r: r["seq"])
        for previous, following in zip(chunks, chunks[1:]):
            pairs += 1
            if following["start"] < previous["end"]:
                with_overlap += 1
                piece = text_by_doc[doc_id][following["start"]:previous["end"]]
                overlap_tokens.append(len(encoder.encode(piece)))
    median_tokens = int(statistics.median(overlap_tokens)) if overlap_tokens else 0
    return with_overlap, pairs, median_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=params.CORPUS_PATH)
    ap.add_argument("--out", type=Path, default=params.CHUNKS_PATH)
    ap.add_argument("--manifest", type=Path, default=params.CHUNKS_MANIFEST_PATH)
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"missing {args.corpus}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    docs = read_jsonl(args.corpus)
    if not docs:
        print(f"empty corpus: {args.corpus}", file=sys.stderr)
        print("aborted: nothing written", file=sys.stderr)
        return 1

    chunker = DocumentChunker(
        chunk_size=params.CHUNK_SIZE,
        chunk_overlap=params.CHUNK_OVERLAP,
        encoding=params.ENCODING,
        delim=params.DELIM,
        include_delim=params.INCLUDE_DELIM,
    )
    rows = chunk_corpus(docs, chunker)
    problems = find_problems(rows, docs)
    if problems:
        for name, examples in problems.items():
            print(f"FAILED {name}: {len(examples)} (e.g. {examples[:5]})", file=sys.stderr)
        print("aborted: nothing written", file=sys.stderr)
        return 1

    with_overlap, pairs, median_tokens = overlap_stats(rows, docs, params.ENCODING)
    tokens = sorted(row["tokens"] for row in rows)
    per_doc = sorted(Counter(row["doc_id"] for row in rows).values())
    elapsed = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": sha256_of(args.corpus),
        "chunk_size": params.CHUNK_SIZE,
        "chunk_overlap": params.CHUNK_OVERLAP,
        "encoding": params.ENCODING,
        "delim": params.DELIM,
        "include_delim": params.INCLUDE_DELIM,
        "docs": len(docs),
        "chunks": len(rows),
        "chunks_per_doc": {"min": per_doc[0], "median": statistics.median(per_doc), "max": per_doc[-1]},
        "tokens_per_chunk": {"min": tokens[0], "median": statistics.median(tokens), "max": tokens[-1]},
        "total_tokens": sum(tokens),
        "overlap_pairs": [with_overlap, pairs],
        "overlap_tokens_median": median_tokens,
        "theme_chunks": dict(Counter(row["theme"] for row in rows)),
        "elapsed_sec": round(elapsed, 2),
        "python_version": platform.python_version(),
        "chonkie_version": version("chonkie"),
        "tiktoken_version": version("tiktoken"),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"docs={len(docs)}  chunks={len(rows)}  elapsed={elapsed:.2f}s")
    print(f"chunks/doc min/median/max = {per_doc[0]} / {statistics.median(per_doc)} / {per_doc[-1]}")
    print(f"tokens/chunk min/median/max = {tokens[0]} / {statistics.median(tokens)} / {tokens[-1]}")
    print(f"overlap pairs = {with_overlap}/{pairs}  median tokens = {median_tokens}")
    print(f"wrote {args.out} and {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
