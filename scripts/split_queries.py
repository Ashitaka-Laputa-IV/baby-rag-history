"""Stratified retrospective dev/test split of the 30-question core set.

This is a robustness check, not a true holdout: the questions were already
used during development. The split is still useful for reporting a smaller,
balanced test slice honestly.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from babyrag.config import EVAL

TARGET = {"single": 4, "entity": 2, "multi": 2, "global": 1, "none": 1}


def main() -> int:
    rows = [json.loads(line) for line in (EVAL / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    rng = random.Random(11)
    test, dev = [], []
    for qtype, items in by_type.items():
        items = list(items)
        rng.shuffle(items)
        n = TARGET.get(qtype, 1)
        test.extend(items[:n])
        dev.extend(items[n:])
    for name, subset in (("queries-test.jsonl", test), ("queries-dev.jsonl", dev)):
        path = EVAL / name
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n",
                        encoding="utf-8")
        print(f"{name}: {len(subset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
