"""Evaluation harness: evidence recall, correctness, abstention, cost."""

from __future__ import annotations

import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import (EVAL, PRICE_INPUT, PRICE_OUTPUT, QUERIES, RESULTS, Settings)
from .graph import build_graph, run_one
from .llm import LLMClient
from .prompts import JUDGE, parse_score
from .retrieval import Services
from .rungs import ALL_RUNGS, LABELS
from .types import add_cost

_local = threading.local()


def _env(settings: Settings):
    if not hasattr(_local, "env"):
        services = Services(settings)
        llm = LLMClient(settings)
        graph = build_graph(services, llm)
        _local.env = (services, llm, graph)
    return _local.env


def read_queries(path: Path = QUERIES) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _norm(text: str) -> str:
    return "".join(str(text).split())


def evidence_in(text: str, evidence: str) -> bool:
    """True only when the gold evidence appears inside the chunk/summary."""
    a, b = _norm(text), _norm(evidence)
    if not b:
        return False
    return b in a or (len(b) >= 12 and b[:12] in a)


def judge(settings: Settings, question: str, gold: str, answer: str) -> int:
    services, llm, _ = _env(settings)
    result = llm.invoke(JUDGE.format(question=question, gold=gold, answer=answer),
                        stage="judge", system="你是严格但公正的评测员。")
    return parse_score(result.text)


def evaluate(settings: Settings, *, rungs: list[str] | None = None,
             limit: int = 0, workers: int = 4, queries_path: Path | None = None,
             run_name: str | None = None) -> list[dict]:
    # Create the cache DB/table once, before worker threads race on it.
    from .cache import Cache
    from .config import CACHE_DB

    Cache(CACHE_DB).close()
    rungs = rungs or ALL_RUNGS
    queries = read_queries(queries_path or QUERIES)
    if limit:
        queries = queries[:limit]
    rows: list[dict] = []
    for rung in rungs:
        print(f"[{rung}] {LABELS.get(rung, rung)}: {len(queries)} questions")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for q in queries:
                mode = "adaptive" if rung == "adaptive" else "ladder"
                futures[pool.submit(_answer_one, settings, q, rung, mode)] = q
            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # one bad question must not stop the run
                    result = {"qid": q["qid"], "rung": rung, "question": q["question"],
                              "answer": "", "abstain": False, "error": str(exc),
                              "cost": {"usd": 0.0, "latency_ms": 0, "llm_calls": 0,
                                       "tokens_in": 0, "tokens_out": 0},
                              "hits": [], "citations": []}
                result["type"] = q.get("type", "")
                result.update(_score(settings, q, result))
                rows.append(result)
                if result.get("error"):
                    print(f"\n  {result['qid']} ERROR: {result['error']}")
                else:
                    print(f"  {result['qid']} OK", end="\r", flush=True)
        print()
    return rows


def _answer_one(settings: Settings, q: dict, rung: str, mode: str) -> dict:
    _, _, graph = _env(settings)
    return run_one(graph, question=q["question"], rung=rung, mode=mode,
                   qid=q["qid"], qtype=q.get("type", ""))


def _score(settings: Settings, q: dict, result: dict) -> dict:
    hits = result.get("hits") or []
    gold_docs = set(q.get("gold_doc_ids") or [])
    evidence = q.get("evidence", "")
    doc_ids = [set(h.get("doc_ids") or [h.get("doc_id", "")]) for h in hits]
    out: dict = {}
    for k in (1, 3, 5, 10):
        out[f"doc_recall@{k}"] = int(any(gold_docs & ids for ids in doc_ids[:k])) if gold_docs else None

    doc_rank = next((i for i, ids in enumerate(doc_ids, start=1) if gold_docs & ids), None) if gold_docs else None
    chunk_rank = next((i for i, h in enumerate(hits, start=1)
                       if evidence_in(h.get("text", ""), evidence)), None) if evidence else None
    out["doc@1"] = int(doc_rank == 1) if doc_rank else (0 if gold_docs else None)
    out["chunk@1"] = int(chunk_rank == 1) if evidence else None
    out["chunk@3"] = int(chunk_rank is not None and chunk_rank <= 3) if evidence else None
    out["chunk@5"] = int(chunk_rank is not None and chunk_rank <= 5) if evidence else None
    out["mrr@10"] = round(1.0 / chunk_rank, 4) if chunk_rank and chunk_rank <= 10 else (0.0 if evidence else None)
    out["ndcg@10"] = round(1.0 / __import__("math").log2(chunk_rank + 1), 4) if chunk_rank and chunk_rank <= 10 else (0.0 if evidence else None)
    out["chunk_hit@5"] = out["chunk@5"]
    out["citation_precision"] = (
        sum(1 for c in result.get("citations", []) if any(c == h.get("chunk_id") for h in hits)) /
        len(result.get("citations", [])) if result.get("citations") else None
    )
    should_abstain = bool(q.get("should_abstain"))
    abstain = bool(result.get("abstain"))
    can_abstain = result.get("rung") in {"r7_crag", "r8_agentic", "adaptive"}
    if should_abstain:
        # A rung without an abstain path is N/A here, not 0.
        out["correct"] = int(abstain) if can_abstain else None
    elif abstain or not result.get("answer"):
        out["correct"] = 0
    else:
        try:
            out["correct"] = judge(settings, q["question"], q.get("gold_answer", ""),
                                   result.get("answer", ""))
        except Exception:
            out["correct"] = None
    return out


def summarize(rows: list[dict]) -> dict:
    by_rung: dict[str, list[dict]] = {}
    for row in rows:
        by_rung.setdefault(row["rung"], []).append(row)
    out: dict = {}
    for rung, items in by_rung.items():
        costs = [float(r.get("cost", {}).get("usd", 0)) for r in items]
        latencies = sorted(int(r.get("cost", {}).get("latency_ms", 0)) for r in items)
        correct = [r["correct"] for r in items if r.get("correct") is not None]
        out[rung] = {
            "n": len(items),
            "correct": sum(correct),
            "correct_n": len(correct),
            "accuracy": round(sum(correct) / len(correct), 4) if correct else None,
            "doc@1": _mean(items, "doc@1"),
            "doc_recall@5": _mean(items, "doc_recall@5"),
            "chunk@1": _mean(items, "chunk@1"),
            "chunk@3": _mean(items, "chunk@3"),
            "chunk@5": _mean(items, "chunk@5"),
            "chunk_hit@5": _mean(items, "chunk@5"),
            "mrr@10": _mean(items, "mrr@10"),
            "ndcg@10": _mean(items, "ndcg@10"),
            "abstain_accuracy": _abstain_accuracy(items),
            "usd_total": round(sum(costs), 6),
            "usd_per_query": round(sum(costs) / len(items), 6) if items else 0,
            "usd_cold_total": round(sum(
                int(r.get("cost", {}).get("tokens_in", 0)) * PRICE_INPUT +
                int(r.get("cost", {}).get("tokens_out", 0)) * PRICE_OUTPUT
                for r in items), 6),
            "usd_cold_per_query": round(sum(
                int(r.get("cost", {}).get("tokens_in", 0)) * PRICE_INPUT +
                int(r.get("cost", {}).get("tokens_out", 0)) * PRICE_OUTPUT
                for r in items) / len(items), 6) if items else 0,
            "p50_ms": latencies[len(latencies) // 2] if latencies else 0,
            "p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0,
            "tokens_in": sum(int(r.get("cost", {}).get("tokens_in", 0)) for r in items),
            "tokens_out": sum(int(r.get("cost", {}).get("tokens_out", 0)) for r in items),
            "llm_calls": sum(int(r.get("cost", {}).get("llm_calls", 0)) for r in items),
        }
        entry = out[rung]
        # Composite: 50% answer accuracy, 30% first-hit retrieval, 20% ranking,
        # with a mild cost penalty so a 3x-expensive rung cannot win a tie.
        quality = (0.50 * (entry["accuracy"] or 0.0)
                   + 0.30 * (entry["chunk@1"] or 0.0)
                   + 0.20 * (entry["mrr@10"] or 0.0))
        factor = 1.0 / (1.0 + 100.0 * (entry["usd_cold_per_query"] or 0.0))
        entry["quality"] = round(quality, 4)
        entry["cost_factor"] = round(factor, 4)
        entry["composite"] = round(100.0 * quality * factor, 1)
    return out


def _mean(items: list[dict], key: str) -> float | None:
    values = [r[key] for r in items if r.get(key) is not None]
    return round(statistics.mean(values), 4) if values else None


def _abstain_accuracy(items: list[dict]) -> float | None:
    if not items or items[0].get("rung") not in {"r7_crag", "r8_agentic", "adaptive"}:
        return None
    subset = [r for r in items if r.get("type") == "none"]
    if not subset:
        return None
    return round(sum(int(r.get("abstain", False)) for r in subset) / len(subset), 4)


def write_reports(rows: list[dict], summary: dict, *, out_dir: Path = RESULTS,
                  run_label: str = "leaderboard", dataset_note: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            slim = dict(row)
            slim["hits"] = [{k: v for k, v in h.items() if k != "text"}
                            for h in row.get("hits", [])]
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# baby-rag-history · {run_label}", "",
             "同一语料、同一批问题；`N/A` 表示该档不具备该能力，不是 0 分。", "",
             "| rung | 里程碑 | 综合 | acc | doc@1 | chunk@1 | chunk@3 | chunk@5 | MRR@10 | nDCG@10 | abstain | cold $/q | p50 | p95 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rung, s in summary.items():
        def fmt(v, pct=False):
            if v is None:
                return "N/A"
            return f"{v * 100:.1f}%" if pct else str(v)
        lines.append(
            f"| `{rung}` | {LABELS.get(rung, rung)} | **{s.get('composite', 0):.1f}** | "
            f"{fmt(s['accuracy'], True)} | "
            f"{fmt(s['doc@1'], True)} | {fmt(s['chunk@1'], True)} | {fmt(s['chunk@3'], True)} | "
            f"{fmt(s['chunk@5'], True)} | {fmt(s['mrr@10'])} | {fmt(s['ndcg@10'])} | "
            f"{fmt(s['abstain_accuracy'], True)} | ${s['usd_cold_per_query']:.5f} | "
            f"{s['p50_ms']}ms | {s['p95_ms']}ms |")
    note = dataset_note or ("评测集 30 题（single 12 / entity 6 / multi 6 / global 3 / none 3）；"
                            "1,000 篇 / 4,740 chunks。")
    lines += ["", "## 说明", "",
              f"- {note}",
              "- `cold $/q` 按本次记录的 token × 单价计算，不受缓存影响；`本次实付` 是实际账单。",
              "- 综合分 = 100 × (0.50×acc + 0.30×chunk@1 + 0.20×MRR@10) / (1 + 100×cold $/q)。",
              "- `chunk@1` / `MRR@10` 是主指标；`doc R@5` 在原始语料上会饱和。",
              "- 成本逐条见 `results/cost.jsonl`。"]
    (out_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'leaderboard.md'}")
