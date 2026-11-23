"""Render the composite-score table as a line chart for README + docs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(r"C:\Users\Ashit\Documents\GitHub\baby-rag-history")
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "assets"

RUNS = [
    ("original", RESULTS, "#4C78A8", "o"),
    ("masked", RESULTS / "masked", "#F58518", "s"),
    ("hard 25", RESULTS / "hard", "#54A24B", "^"),
    ("masked + hard 25", RESULTS / "masked-hard", "#B279A2", "D"),
]

SHORT = {
    "r0_dense": "r0", "r1_hybrid": "r1", "r2_rerank": "r2",
    "r3_hyde": "r3", "r4_multiquery": "r4", "r5_contextual": "r5",
    "r6_raptor": "r6", "r7_crag": "r7", "r8_agentic": "r8",
    "adaptive": "adaptive",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {name: json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                 for name, folder, _, _ in RUNS}
    rungs = list(summaries["original"].keys())
    x = list(range(len(rungs)))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    for name, _, color, marker in RUNS:
        values = [summaries[name][r]["composite"] for r in rungs]
        ax.plot(x, values, marker=marker, color=color, linewidth=2.2,
                markersize=7, label=name, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(r, r) for r in rungs])
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel("Composite score (higher is better)")
    ax.set_title("Composite score by rung and evaluation setting", pad=14)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              frameon=False, fontsize=9)

    best = max(rungs, key=lambda r: summaries["original"][r]["composite"])
    best_value = summaries["original"][best]["composite"]
    ax.annotate(f"{SHORT[best]} {best_value:.1f}",
                xy=(rungs.index(best), best_value),
                xytext=(6, 8), textcoords="offset points", fontsize=9,
                color="#4C78A8")

    fig.tight_layout()
    fig.savefig(OUT / "composite-line.png", bbox_inches="tight")
    fig.savefig(OUT / "composite-line.svg", bbox_inches="tight")
    print(f"wrote {OUT / 'composite-line.png'} and .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
