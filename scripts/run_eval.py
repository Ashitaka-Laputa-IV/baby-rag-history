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
    ap.add_argument("--provider", default="opencode", choices=["opencode", "siliconflow"],
                    help="chat model provider; siliconflow uses the free Qwen model")
    ap.add_argument("--model", default=None, help="chat model id for the selected provider")
    ap.add_argument("--variant", default="original", choices=["original", "masked", "public"])
    ap.add_argument("--queries", default=None, help="path to a queries JSONL")
    args = ap.parse_args()
    os.environ["BABYRAG_VARIANT"] = args.variant
    os.environ["BABYRAG_LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["BABYRAG_LLM_MODEL"] = args.model
    chosen = ALL_RUNGS if args.rungs == "all" else [r.strip() for r in args.rungs.split(",")]
    queries_path = Path(args.queries) if args.queries else None
    tag = ""
    if queries_path is not None:
        tag = queries_path.stem.removeprefix("queries-")
        if tag == queries_path.stem:
            tag = ""
    hard = tag == "hard"
    test = tag == "test"
    manual = tag.startswith("manual")
    run_name = None
    if args.provider == "siliconflow":
        slug = (args.model or "Qwen/Qwen2.5-7B-Instruct").replace("/", "-").replace(".", "").lower()
        run_name = f"siliconflow-{slug}"
        if tag:
            run_name += f"-{tag}"
    elif args.variant == "public":
        run_name = "public"
    elif tag:
        run_name = f"{args.variant}-{tag}" if args.variant == "masked" else tag
    elif args.variant == "masked":
        run_name = args.variant
    settings = load_settings()
    rows = evaluate(settings, rungs=chosen, limit=args.limit, workers=args.workers,
                    queries_path=queries_path, run_name=run_name)
    summary = summarize(rows)
    if args.provider == "siliconflow":
        label = f"SiliconFlow {settings.model}"
        if manual:
            note = (f"弱模型对照：{settings.model}（SiliconFlow 免费档）。50 道中段事实手写题；"
                    "成本按 $0 记录，不与 DeepSeek 主榜混算。judge 也是该弱模型，正确率噪声更大。")
        else:
            note = (f"弱模型对照：{settings.model}（SiliconFlow 免费档）。30 道原题；"
                    "成本按 $0 记录，不与 DeepSeek 主榜混算。")
    elif test:
        label = "retrospective test split"
        note = "10 道分层 test 子集（single 4 / entity 2 / multi 2 / global 1 / none 1）；题目在开发期已见过，只能作为稳健性检查，不是真正 holdout。"
    elif args.variant == "public":
        label = "RAGBench external anchor"
        note = "RAGBench emanual test 子集（40 题，英文用户手册）；只作外部锚点，不与中文维基主榜混算。"
    elif manual:
        label = "middle-fact questions (50, handwritten)"
        note = ("50 道手写题，答案均为文档中段的事实；每题 evidence 已校验落在 gold 文档中间三分之一的 chunk 内。"
                "用于单独衡量“中段事实”检索，不替代原 30 题主榜。")
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
