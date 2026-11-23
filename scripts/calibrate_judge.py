"""Sample-30 judge calibration: A/B field-order swap + agreement with stored scores."""

from __future__ import annotations

import json
import random
from pathlib import Path

from babyrag.config import EVAL, RESULTS, load_settings
from babyrag.eval import judge_ab

RUNS = Path(r"C:\Users\Ashit\Documents\GitHub\baby-rag-history\results")


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    settings = load_settings()
    queries = {q["qid"]: q for q in read(EVAL / "queries.jsonl")}
    queries.update({q["qid"]: q for q in read(EVAL / "queries-hard.jsonl")})
    rows = [r for r in read(RUNS / "per_query.jsonl")
            if r.get("type") != "none" and r.get("correct") in (0, 1)]
    positives = [r for r in rows if r["correct"] == 1]
    negatives = [r for r in rows if r["correct"] == 0]
    rng = random.Random(7)
    sample = rng.sample(positives, min(15, len(positives))) + rng.sample(negatives, min(15, len(negatives)))

    out = []
    agree_normal = agree_ab = 0
    for row in sample:
        q = queries[row["qid"]]
        a, b, agree = judge_ab(settings, q["question"], q.get("gold_answer", ""), row.get("answer", ""))
        agree_normal += int(a == row["correct"])
        agree_ab += int(agree)
        out.append({"qid": row["qid"], "stored": row["correct"], "normal": a,
                    "swapped": b, "ab_agree": agree})
        print(f"{row['qid']} stored={row['correct']} normal={a} swapped={b} agree={agree}")

    summary = {
        "n": len(out),
        "normal_agrees_with_stored": agree_normal,
        "ab_order_agree": agree_ab,
        "normal_rate": round(agree_normal / len(out), 4),
        "ab_rate": round(agree_ab / len(out), 4),
        "items": out,
    }
    (RESULTS / "judge-calibration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
