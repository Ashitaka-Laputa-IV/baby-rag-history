"""Recompute faithfulness aggregates from the stored per-answer items."""

from __future__ import annotations

import json
from pathlib import Path

from babyrag.config import RESULTS


def main() -> int:
    path = RESULTS / "faithfulness.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    by_rung: dict[str, dict] = {}
    for row in data["items"]:
        bucket = by_rung.setdefault(row["rung"], {
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
    for bucket in by_rung.values():
        bucket["supported_avg"] = (round(bucket["supported"] / bucket["supported_n"], 3)
                                   if bucket["supported_n"] else None)
        bucket["citation_ok_rate"] = (round(bucket["citation_ok"] / bucket["citation_n"], 3)
                                      if bucket["citation_n"] else None)
    data["by_rung"] = by_rung
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Faithfulness / citation support", "",
             f"Sampled {len(data['items'])} non-abstaining answers (max 3 per rung).", "",
             "| rung | n | supported | citation_ok | unsupported claims |",
             "|---|---:|---:|---:|---:|"]
    for rung, b in sorted(by_rung.items()):
        sup = "N/A" if b["supported_avg"] is None else f"{b['supported_avg']:.2f}"
        cit = "N/A" if b["citation_ok_rate"] is None else f"{b['citation_ok_rate']:.0%}"
        lines.append(f"| `{rung}` | {b['n']} | {sup} | {cit} | {b['unsupported']} |")
    lines += ["", "`supported` is the judge's estimated fraction of factual claims backed by the "
              "numbered context; `citation_ok` is whether citations point at supporting passages.",
              "One answer failed JSON parsing and is excluded from the averages (still counted in n)."]
    (RESULTS / "faithfulness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(by_rung, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
