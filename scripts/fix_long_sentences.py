"""Split over-long sentences so chonkie can respect the 512-token cap.

The rag-ladder open-questions file recommends this (option B): split a sentence
that exceeds the chunk budget at Chinese punctuation, then re-chunk. Only the
few affected docs are rewritten; punctuation is preserved, so evidence spans
and gold doc ids stay valid.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import tiktoken

from babyrag.config import CORPUS, ENCODING

SENTENCE = re.compile(r"[^。！？；\n]*[。！？；\n]?")
SUB = re.compile(r"[^，、：]*[，、：]?")


def split_sentence(sentence: str, enc, limit: int = 400) -> str:
    if len(enc.encode(sentence)) <= limit:
        return sentence
    parts = [p for p in SUB.findall(sentence) if p]
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = buf + part
        if buf and len(enc.encode(candidate)) > limit:
            out.append(buf)
            buf = part
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return "\n".join(out)


def fix(text: str, enc, limit: int = 400) -> tuple[str, int]:
    changed = 0
    pieces: list[str] = []
    for sentence in SENTENCE.findall(text):
        if not sentence:
            continue
        fixed = split_sentence(sentence, enc, limit)
        if fixed != sentence:
            changed += 1
        pieces.append(fixed)
    return "".join(pieces), changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    enc = tiktoken.get_encoding(ENCODING)
    docs = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    total = 0
    for doc in docs:
        fixed, changed = fix(doc["text"], enc, args.limit)
        if changed:
            doc["text"] = fixed
            doc["chars"] = len(fixed)
            total += changed
    with args.corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"docs={len(docs)} long sentences split={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
