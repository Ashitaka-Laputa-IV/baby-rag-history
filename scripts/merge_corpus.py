"""Merge the curated 200-doc seed with a larger rule-classified sample.

The seed is kept first and unchanged so every existing question's gold_doc_id
still resolves. New docs are appended by title, up to `--size`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from babyrag.config import CORPUS, DATA


def norm_title(title: str) -> str:
    return "".join(ch for ch in title.strip().lower() if ch.isalnum())


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=Path, default=DATA / "corpus.base.jsonl")
    ap.add_argument("--extra", type=Path, default=DATA / "corpus.rule.jsonl")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=CORPUS)
    args = ap.parse_args()

    seed = read(args.seed)
    extra = read(args.extra)
    seen = {norm_title(d["title"]) for d in seed}
    merged = list(seed)
    added = 0
    for doc in extra:
        key = norm_title(doc["title"])
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        added += 1
        if len(merged) >= args.size:
            break

    args.out.write_text("", encoding="utf-8")
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in merged:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    counts = Counter(d["theme"] for d in merged)
    print(f"seed={len(seed)} added={added} total={len(merged)}")
    print("themes=" + " ".join(f"{k}:{v}" for k, v in counts.items()))

    queries = read(DATA.parent / "eval" / "queries.jsonl")
    ids = {d["doc_id"] for d in merged}
    missing = {qid for q in queries for qid in q.get("gold_doc_ids", []) if qid not in ids}
    print(f"gold docs missing: {len(missing)}")
    if missing:
        print("  " + ", ".join(sorted(missing)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
