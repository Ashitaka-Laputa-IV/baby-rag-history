"""Reconcile stored results after the fairness audit.

1. Restore the original 300-row run (it was overwritten by a 1-question smoke
   test) and backfill chunk@1/MRR/nDCG from the stored hits.
2. Drop the four unfair hard questions (ambiguous “该市镇/该师团” references)
   from the hard and masked-hard runs, leaving 25 fair questions.
"""

from __future__ import annotations

import json
from pathlib import Path

from babyrag.config import DATA, EVAL, RESULTS
from babyrag.eval import evidence_in, summarize, write_reports

UNFAIR = {"h010", "h011", "h021", "h022"}
ORIG = Path(r"C:\Users\Ashit\AppData\Local\Temp\opencode\orig_pq.jsonl")


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    queries = {q["qid"]: q for q in read(EVAL / "queries.jsonl")}
    chunk_text = {r["chunk_id"]: r["text"] for r in read(DATA / "chunks.jsonl")}
    raptor_text = {r["node_id"]: r["text"] for r in read(DATA / "raptor.jsonl")}

    rows = read(ORIG)
    for row in rows:
        q = queries[row["qid"]]
        row["type"] = q.get("type", "")
        evidence = q.get("evidence", "")
        gold = set(q.get("gold_doc_ids") or [])
        hits = row.get("hits") or []
        doc_rank = next((i for i, h in enumerate(hits, start=1)
                         if gold & set(h.get("doc_ids") or [h.get("doc_id", "")])), None)
        chunk_rank = next((i for i, h in enumerate(hits, start=1)
                           if evidence_in(chunk_text.get(h["chunk_id"], "")
                                          or raptor_text.get(h["chunk_id"], ""), evidence)), None)
        row["doc@1"] = int(doc_rank == 1) if doc_rank else 0
        row["chunk@1"] = int(chunk_rank == 1) if evidence else None
        row["chunk@3"] = int(chunk_rank is not None and chunk_rank <= 3) if evidence else None
        row["chunk@5"] = int(chunk_rank is not None and chunk_rank <= 5) if evidence else None
        row["mrr@10"] = round(1 / chunk_rank, 4) if chunk_rank and chunk_rank <= 10 else 0.0
        row["ndcg@10"] = round(1 / __import__("math").log2(chunk_rank + 1), 4) if chunk_rank and chunk_rank <= 10 else 0.0
        row["chunk_hit@5"] = row["chunk@5"]
    write(RESULTS / "per_query.jsonl", rows)
    write_reports(rows, summarize(rows), out_dir=RESULTS, run_label="leaderboard",
                  dataset_note="30 道原题；1,000 篇 / 4,740 chunks。")
    print("original restored and backfilled")

    for name, label, note in (
        ("hard", "hard questions (25 fair)",
         "25 道公平手写难题（已剔除 4 道指代不清的题）；避开文档标题，保留年份/地区/职业等限定词。"),
        ("masked-hard", "masked + hard (25 fair)",
         "25 道公平手写难题 × 去标题 + 干扰文档语料（1,200 篇 / 5,676 chunks）。"),
    ):
        folder = RESULTS / name
        rows = [r for r in read(folder / "per_query.jsonl") if r["qid"] not in UNFAIR]
        write(folder / "per_query.jsonl", rows)
        summary = summarize(rows)
        write_reports(rows, summary, out_dir=folder, run_label=label, dataset_note=note)
        print(f"{name}: {len(rows)} rows")
        for rung, s in summary.items():
            print(f"  {rung:14} acc={s['accuracy']} chunk@1={s['chunk@1']} mrr={s['mrr@10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
