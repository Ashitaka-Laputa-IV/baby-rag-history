"""Sample answers and judge faithfulness / citation support.

For each sampled answer we rebuild the numbered context the generator saw,
then ask one strict judge whether the answer's factual claims are supported by
the cited passages. This is deliberately a sample, not a full re-judge.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from babyrag.config import DATA, EVAL, RESULTS, load_settings
from babyrag.llm import LLMClient
from babyrag.prompts import FAITHFULNESS

RUN = RESULTS


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"supported": None, "citation_ok": None, "unsupported": ["<bad json>"]}
    try:
        body = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"supported": None, "citation_ok": None, "unsupported": ["<bad json>"]}
    supported = body.get("supported")
    try:
        supported = float(supported) if supported is not None else None
    except (TypeError, ValueError):
        supported = None
    return {"supported": supported,
            "citation_ok": body.get("citation_ok"),
            "unsupported": body.get("unsupported") or []}


def main() -> int:
    settings = load_settings()
    llm = LLMClient(settings, max_output_tokens=500)
    queries = {q["qid"]: q for q in read(EVAL / "queries.jsonl")}
    chunk_text = {r["chunk_id"]: r["text"] for r in read(DATA / "chunks.jsonl")}
    raptor_text = {r["node_id"]: r["text"] for r in read(DATA / "raptor.jsonl")}
    rows = [r for r in read(RUN / "per_query.jsonl") if not r.get("abstain") and r.get("answer")]

    by_rung: dict[str, list[dict]] = {}
    for row in rows:
        by_rung.setdefault(row["rung"], []).append(row)
    rng = random.Random(7)
    sample: list[dict] = []
    for rung in sorted(by_rung):
        items = by_rung[rung]
        sample.extend(rng.sample(items, min(3, len(items))))

    results = []
    for row in sample:
        q = queries[row["qid"]]
        pieces = []
        for i, hit in enumerate(row.get("hits") or [], start=1):
            text = chunk_text.get(hit["chunk_id"]) or raptor_text.get(hit["chunk_id"], "")
            pieces.append(f"[{i}] {text[:900]}")
        context = "\n\n".join(pieces)
        prompt = FAITHFULNESS.format(question=q["question"], answer=row["answer"],
                                     context=context)
        reply = llm.invoke(prompt, stage="faithfulness", prompt_version="v1",
                           system="你只输出合法 JSON。")
        judged = parse(reply.text)
        results.append({"qid": row["qid"], "rung": row["rung"],
                        "answer": row["answer"][:300], **judged})
        print(f"{row['rung']:14} {row['qid']} supported={judged['supported']} "
              f"citation_ok={judged['citation_ok']}")

    by_rung_out: dict[str, dict] = {}
    for row in results:
        bucket = by_rung_out.setdefault(row["rung"], {
            "n": 0, "supported_n": 0, "supported": 0.0,
            "citation_n": 0, "citation_ok": 0, "unsupported": 0})
        bucket["n"] += 1
        if row["supported"] is not None:
            bucket["supported"] += row["supported"]
            bucket["supported_n"] += 1
        if row["citation_ok"] is not None:
            bucket["citation_n"] += 1
            bucket["citation_ok"] += int(bool(row["citation_ok"]))
        bucket["unsupported"] += len(row["unsupported"])
    for bucket in by_rung_out.values():
        bucket["supported_avg"] = (round(bucket["supported"] / bucket["supported_n"], 3)
                                   if bucket["supported_n"] else None)
        bucket["citation_ok_rate"] = (round(bucket["citation_ok"] / bucket["citation_n"], 3)
                                      if bucket["citation_n"] else None)

    payload = {"sampled_answers": len(results), "by_rung": by_rung_out, "items": results}
    (RESULTS / "faithfulness.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Faithfulness / citation support", "",
             f"Sampled {len(results)} non-abstaining answers (max 3 per rung).", "",
             "| rung | n | supported | citation_ok | unsupported claims |",
             "|---|---:|---:|---:|---:|"]
    for rung, b in sorted(by_rung_out.items()):
        supported = "N/A" if b["supported_avg"] is None else f"{b['supported_avg']:.2f}"
        citation = "N/A" if b["citation_ok_rate"] is None else f"{b['citation_ok_rate']:.0%}"
        lines.append(f"| `{rung}` | {b['n']} | {supported} | {citation} | {b['unsupported']} |")
    lines += ["", "`supported` is the judge's estimated fraction of factual claims backed by the "
              "numbered context; `citation_ok` is whether citations point at supporting passages."]
    (RESULTS / "faithfulness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS / 'faithfulness.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
