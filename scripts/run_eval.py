"""Run the ladder from the command line: `uv run python -m scripts.run_eval`."""

from __future__ import annotations

import argparse

from babyrag.config import load_settings
from babyrag.eval import evaluate, summarize, write_reports
from babyrag.rungs import ALL_RUNGS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    chosen = ALL_RUNGS if args.rungs == "all" else [r.strip() for r in args.rungs.split(",")]
    settings = load_settings()
    rows = evaluate(settings, rungs=chosen, limit=args.limit, workers=args.workers)
    summary = summarize(rows)
    write_reports(rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
