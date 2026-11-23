"""Evaluation harness: evidence recall, correctness, abstention, cost."""

from __future__ import annotations

import json
import math
import random
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import (EVAL, PRICE_INPUT, PRICE_OUTPUT, QUERIES, RESULTS, Settings)
from .graph import build_graph, run_one
from .llm import LLMClient
from .prompts import JUDGE, JUDGE_SWAP, parse_score
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


def judge_ab(settings: Settings, question: str, gold: str, answer: str) -> tuple[int, int, bool]:
    """Two judgments with the answer/gold fields reversed; agreement is a
    cheap position-bias check, not a replacement for human calibration."""
    services, llm, _ = _env(settings)
    first = llm.invoke(JUDGE.format(question=question, gold=gold, answer=answer),
                       stage="judge", system="你是严格但公正的评测员。")
    second = llm.invoke(JUDGE_SWAP.format(question=question, gold=gold, answer=answer),
                        stage="judge_swap", system="你是严格但公正的评测员。")
    a, b = parse_score(first.text), parse_score(second.text)
    return a, b, a == b


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

    def relevant(h: dict) -> bool:
        # RAPTOR summary nodes are legitimate abstractions: count them as a
        # retrieval hit when the summary covers a gold document, instead of
        # requiring the raw evidence string inside the summary.
        if str(h.get("chunk_id", "")).startswith("raptor:"):
            return bool(gold_docs & set(h.get("doc_ids") or []))
        return evidence_in(h.get("text", ""), evidence)

    chunk_rank = next((i for i, h in enumerate(hits, start=1) if relevant(h)), None) if evidence else None
    out["doc@1"] = int(doc_rank == 1) if doc_rank else (0 if gold_docs else None)
    top_docs = set()
    for h in hits[:5]:
        top_docs.update(h.get("doc_ids") or [h.get("doc_id", "")])
    out["doc_recall_all@5"] = (int(bool(gold_docs) and gold_docs.issubset(top_docs))
                               if gold_docs else None)
    out["context_precision@5"] = (
        round(sum(1 for h in hits[:5] if relevant(h)) / len(hits[:5]), 4)
        if hits and evidence else None
    )
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
    if should_abstain:
        # No-answer questions never enter the answerable accuracy denominator.
        # Every rung is scored on abstention, including rungs that cannot abstain
        # (they score 0 there, which is the honest number).
        out["correct"] = None
        out["abstain_correct"] = int(abstain)
    else:
        out["abstain_correct"] = None
        if abstain or not result.get("answer"):
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
            "abstain_n": sum(1 for r in items if r.get("abstain_correct") is not None),
            "accuracy": round(sum(correct) / len(correct), 4) if correct else None,
            "accuracy_ci": _bootstrap_ci([float(x) for x in correct]),
            "accuracy_ci_wilson": _wilson_ci([float(x) for x in correct]),
            "doc@1": _mean(items, "doc@1"),
            "doc_recall@5": _mean(items, "doc_recall@5"),
            "doc_recall_all@5": _mean(items, "doc_recall_all@5"),
            "context_precision@5": _mean(items, "context_precision@5"),
            "chunk@1": _mean(items, "chunk@1"),
            "chunk@1_ci": _bootstrap_ci([float(r["chunk@1"]) for r in items
                                         if r.get("chunk@1") is not None]),
            "chunk@1_ci_wilson": _wilson_ci([float(r["chunk@1"]) for r in items
                                             if r.get("chunk@1") is not None]),
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
        entry["abstain_ci"] = _bootstrap_ci([float(r["abstain_correct"]) for r in items
                                             if r.get("abstain_correct") is not None])
        entry["abstain_ci_wilson"] = _wilson_ci([float(r["abstain_correct"]) for r in items
                                                 if r.get("abstain_correct") is not None])
        # Composite: 50% answerable-question accuracy, 30% first-hit retrieval,
        # 20% ranking, with a mild cost penalty so a 3x-expensive rung cannot
        # win a tie. Abstention is reported separately, not folded in.
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
    values = [r["abstain_correct"] for r in items if r.get("abstain_correct") is not None]
    return round(statistics.mean(values), 4) if values else None


def _bootstrap_ci(values: list[float], *, n: int = 1000, seed: int = 42) -> tuple[float | None, float | None]:
    """Percentile bootstrap 95% CI; deterministic for a fixed seed."""
    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return (round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4))


def _wilson_ci(values: list[float]) -> tuple[float | None, float | None]:
    """Wilson score 95% CI for a proportion; stable at 0/0 and n/n."""
    n = len(values)
    if n == 0:
        return (None, None)
    p = sum(values) / n
    z = 1.96
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


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
             "| rung | 里程碑 | 综合 | 可答 acc (95% CI) | 弃答 acc (95% CI) | doc@1 | doc all@5 | evidence R@1 (95% CI) | R@3 | R@5 | MRR@10 | nDCG@10 | cold $/q | p50 | p95 |",
             "|---|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rung, s in summary.items():
        def fmt(v, pct=False):
            if v is None:
                return "N/A"
            return f"{v * 100:.1f}%" if pct else str(v)

        def with_ci(v, pair, pct=True):
            base = fmt(v, pct)
            if pair is None or pair[0] is None:
                return base
            if pct:
                return f"{base} [{pair[0] * 100:.0f}–{pair[1] * 100:.0f}]"
            return f"{base} [{pair[0]:.2f}–{pair[1]:.2f}]"

        lines.append(
            f"| `{rung}` | {LABELS.get(rung, rung)} | **{s.get('composite', 0):.1f}** | "
            f"{with_ci(s['accuracy'], s.get('accuracy_ci_wilson'), True)} | "
            f"{with_ci(s['abstain_accuracy'], s.get('abstain_ci_wilson'), True)} | "
            f"{fmt(s['doc@1'], True)} | {fmt(s.get('doc_recall_all@5'), True)} | "
            f"{with_ci(s['chunk@1'], s.get('chunk@1_ci_wilson'), True)} | "
            f"{fmt(s['chunk@3'], True)} | {fmt(s['chunk@5'], True)} | "
            f"{fmt(s['mrr@10'])} | {fmt(s['ndcg@10'])} | "
            f"${s['usd_cold_per_query']:.5f} | "
            f"{s['p50_ms']}ms | {s['p95_ms']}ms |")
    note = dataset_note or ("评测集 30 题（single 12 / entity 6 / multi 6 / global 3 / none 3）；"
                            "1,000 篇 / 4,740 chunks。")
    lines += ["", "## 说明", "",
              f"- {note}",
              "- `可答 acc` 只算可回答问题（所有 rung 同一分母）；`弃答 acc` 只算无答案题，所有 rung 都报。",
              "- 方括号是 Wilson 95% 置信区间（小样本比例比 bootstrap 更稳）；bootstrap 结果仍在 `summary.json`。",
              "- 综合分 = 100 × (0.50×可答 acc + 0.30×evidence R@1 + 0.20×MRR@10) / (1 + 100×cold $/q)。",
              "- `evidence R@k` = gold evidence 是否进入最终上下文；`doc all@5` 要求所有 gold 文档都进 top-5（多跳更严）。",
              "- RAPTOR 摘要节点按「覆盖 gold 文档」算命中，不要求摘要逐字包含 evidence。",
              "- `evidence R@1` / `MRR@10` 是主指标；`doc R@5` 在原始语料上会饱和。",
              "- 成本逐条见 `results/cost.jsonl`。"]
    (out_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'leaderboard.md'}")
