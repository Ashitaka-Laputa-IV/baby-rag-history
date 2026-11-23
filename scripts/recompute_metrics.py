"""Recompute stored results under the fixed evaluation protocol.

- Answerable accuracy and abstention accuracy now share one denominator.
- RAPTOR summary nodes count as retrieval hits when they cover a gold doc.
- Adds bootstrap 95% CIs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from babyrag.config import DATA, EVAL, RESULTS
from babyrag.eval import evidence_in, summarize, write_reports

UNFAIR = {"h010", "h011", "h021", "h022"}
RUNS = {
    "": ("leaderboard", "30 道原题；1,000 篇 / 4,740 chunks。", False),
    "masked": ("masked benchmark",
               "30 道原题，chunk 标题替换为「（标题已隐藏）」，另加 200 篇同姓/同类型干扰文档（1,200 篇 / 5,676 chunks）。",
               False),
    "hard": ("hard questions (25 fair)",
             "25 道公平手写难题（已剔除 4 道指代不清的题）；避开文档标题，保留年份/地区/职业等限定词。",
             True),
    "masked-hard": ("masked + hard (25 fair)",
                    "25 道公平手写难题 × 去标题 + 干扰文档语料（1,200 篇 / 5,676 chunks）。",
                    True),
    "test": ("retrospective test split",
             "10 道分层 test 子集（single 4 / entity 2 / multi 2 / global 1 / none 1）；题目在开发期已见过，只能作为稳健性检查，不是真正 holdout。",
             False),
}


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    queries = {q["qid"]: q for q in read(EVAL / "queries.jsonl")}
    queries.update({q["qid"]: q for q in read(EVAL / "queries-hard.jsonl")})
    queries.update({q["qid"]: q for q in read(EVAL / "queries-test.jsonl")})
    chunk_texts = {
        "original": {r["chunk_id"]: r["text"] for r in read(DATA / "chunks.jsonl")},
        "masked": {r["chunk_id"]: r["text"] for r in read(DATA / "chunks.masked.jsonl")},
    }
    raptor_texts = {
        "original": {r["node_id"]: r["text"] for r in read(DATA / "raptor.jsonl")},
        "masked": {r["node_id"]: r["text"] for r in read(DATA / "raptor.masked.jsonl")},
    }

    for name, (label, note, hard) in RUNS.items():
        folder = RESULTS / name if name else RESULTS
        path = folder / "per_query.jsonl"
        if not path.exists():
            print(f"skip {name}: no per_query.jsonl")
            continue
        rows = read(path)
        if hard:
            rows = [r for r in rows if r["qid"] not in UNFAIR]
        variant = "masked" if name.startswith("masked") else "original"
        chunk_text = chunk_texts[variant]
        raptor_text = raptor_texts[variant]

        for row in rows:
            q = queries[row["qid"]]
            row["type"] = q.get("type", "")
            evidence = q.get("evidence", "")
            gold = set(q.get("gold_doc_ids") or [])
            if q.get("should_abstain"):
                row["abstain_correct"] = int(bool(row.get("abstain")))
                row["correct"] = None
            else:
                row["abstain_correct"] = None

            hits = row.get("hits") or []

            def relevant(h: dict) -> bool:
                if str(h.get("chunk_id", "")).startswith("raptor:"):
                    return bool(gold & set(h.get("doc_ids") or []))
                text = chunk_text.get(h.get("chunk_id", "")) or raptor_text.get(h.get("chunk_id", ""), "")
                return evidence_in(text, evidence)

            doc_rank = next((i for i, h in enumerate(hits, start=1)
                             if gold & set(h.get("doc_ids") or [h.get("doc_id", "")])), None)
            chunk_rank = next((i for i, h in enumerate(hits, start=1) if relevant(h)), None) if evidence else None
            row["doc@1"] = int(doc_rank == 1) if doc_rank else (0 if gold else None)
            top_docs = set()
            for h in hits[:5]:
                top_docs.update(h.get("doc_ids") or [h.get("doc_id", "")])
            row["doc_recall_all@5"] = (int(bool(gold) and gold.issubset(top_docs))
                                       if gold else None)
            row["context_precision@5"] = (
                round(sum(1 for h in hits[:5] if relevant(h)) / len(hits[:5]), 4)
                if hits and evidence else None
            )
            row["chunk@1"] = int(chunk_rank == 1) if evidence else None
            row["chunk@3"] = int(chunk_rank is not None and chunk_rank <= 3) if evidence else None
            row["chunk@5"] = int(chunk_rank is not None and chunk_rank <= 5) if evidence else None
            row["mrr@10"] = round(1 / chunk_rank, 4) if chunk_rank and chunk_rank <= 10 else 0.0
            row["ndcg@10"] = round(1 / math.log2(chunk_rank + 1), 4) if chunk_rank and chunk_rank <= 10 else 0.0
            row["chunk_hit@5"] = row["chunk@5"]

        write(path, rows)
        summary = summarize(rows)
        write_reports(rows, summary, out_dir=folder, run_label=label, dataset_note=note)
        print(f"{name or 'original'}: {len(rows)} rows")
        for rung, s in summary.items():
            print(f"  {rung:14} acc={s['accuracy']} ci={s['accuracy_ci']} "
                  f"chunk@1={s['chunk@1']} abstain={s['abstain_accuracy']} composite={s['composite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
