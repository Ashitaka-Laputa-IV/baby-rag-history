"""Run the ladder from the command line: `uv run python -m scripts.run_eval`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from babyrag.config import RESULTS, load_settings
from babyrag.eval import evaluate, summarize, write_reports
from babyrag.rungs import ALL_RUNGS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variant", default="original", choices=["original", "masked", "public"])
    ap.add_argument("--queries", default=None, help="path to a queries JSONL")
    args = ap.parse_args()
    os.environ["BABYRAG_VARIANT"] = args.variant
    chosen = ALL_RUNGS if args.rungs == "all" else [r.strip() for r in args.rungs.split(",")]
    queries_path = Path(args.queries) if args.queries else None
    hard = queries_path is not None and "hard" in queries_path.name
    test = queries_path is not None and "test" in queries_path.name
    run_name = None
    if args.variant == "masked" or args.variant == "public" or hard or test:
        if test:
            run_name = "test"
        elif hard:
            run_name = f"{args.variant}-hard" if args.variant == "masked" else "hard"
        else:
            run_name = args.variant
    settings = load_settings()
    rows = evaluate(settings, rungs=chosen, limit=args.limit, workers=args.workers,
                    queries_path=queries_path, run_name=run_name)
    summary = summarize(rows)
    if test:
        label = "retrospective test split"
        note = "10 道分层 test 子集（single 4 / entity 2 / multi 2 / global 1 / none 1）；题目在开发期已见过，只能作为稳健性检查，不是真正 holdout。"
    elif args.variant == "public":
        label = "RAGBench external anchor"
        note = "RAGBench emanual test 子集（40 题，英文用户手册）；只作外部锚点，不与中文维基主榜混算。"
    elif hard:
        label = "masked + hard questions" if args.variant == "masked" else "hard questions"
        note = "29 道手写难题：避开文档标题，保留年份/地区/职业等限定词；每题 evidence 已校验落在 gold 文档内。"
    elif args.variant == "masked":
        label = "masked benchmark"
        note = "30 道原题，chunk 标题替换为「（标题已隐藏）」，并加入 200 篇同姓/同类型干扰文档（1,200 篇 / 5,676 chunks）。"
    else:
        label = "leaderboard"
        note = ""
    write_reports(rows, summary, out_dir=RESULTS / run_name if run_name else RESULTS,
                  run_label=label, dataset_note=note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
