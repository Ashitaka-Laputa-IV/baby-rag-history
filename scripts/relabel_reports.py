"""Rebuild report headers and notes from existing per_query.jsonl files."""

from __future__ import annotations

import json
from pathlib import Path

from babyrag.config import RESULTS
from babyrag.eval import summarize, write_reports

RUNS = {
    "": ("leaderboard", "30 道原题；1,000 篇 / 4,740 chunks。"),
    "masked": ("masked benchmark",
               "30 道原题，chunk 标题替换为「（标题已隐藏）」，另加 200 篇同姓/同类型干扰文档（1,200 篇 / 5,676 chunks）。"),
    "hard": ("hard questions",
             "29 道手写难题：避开文档标题，保留年份/地区/职业等限定词；evidence 已校验落在 gold 文档内。"),
    "masked-hard": ("masked + hard questions",
                    "29 道手写难题 × 去标题 + 干扰文档语料（1,200 篇 / 5,676 chunks）。"),
}


def main() -> int:
    for name, (label, note) in RUNS.items():
        folder = RESULTS / name if name else RESULTS
        path = folder / "per_query.jsonl"
        if not path.exists():
            print(f"skip {name or 'original'}: no per_query.jsonl")
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        write_reports(rows, summarize(rows), out_dir=folder,
                      run_label=label, dataset_note=note)
        print(f"relabelled {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
