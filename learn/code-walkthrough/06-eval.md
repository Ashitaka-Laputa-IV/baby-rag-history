---
title: 代码漫游 06 · 评测
date: 2026-11-24
input_shape: concept
subject: eval.py 逐函数阅读
---

# 代码漫游 06 · 评测

这一篇读 [`eval.py`](../../src/babyrag/eval.py)。它是“批量跑题 + 打分 + 出报告”。

## 0. 结构

```
_env               # 线程本地 Services/LLM/graph
read_queries
_norm / evidence_in
judge / judge_ab
evaluate           # 双循环：rung × query
_answer_one
_score             # 单题所有指标
summarize          # 按 rung 汇总 + CI + 综合分
_mean / _abstain_accuracy / _bootstrap_ci / _wilson_ci
write_reports      # per_query.jsonl / summary.json / leaderboard.md
```

## 1. 线程本地环境

[`eval.py:24`](../../src/babyrag/eval.py#L24)：

```python
_local = threading.local()

def _env(settings):
    if not hasattr(_local, "env"):
        services = Services(settings)
        llm = LLMClient(settings)
        graph = build_graph(services, llm)
        _local.env = (services, llm, graph)
    return _local.env
```

每个线程一套 Chroma/SQLite/graph，避免跨线程共享。这也是为什么 `evaluate` 可以先
`Cache(CACHE_DB).close()` 建表，再开线程池。

## 2. `evidence_in`

[`eval.py:42`](../../src/babyrag/eval.py#L42)：

```python
def evidence_in(text: str, evidence: str) -> bool:
    a, b = _norm(text), _norm(evidence)
    if not b:
        return False
    return b in a or (len(b) >= 12 and b[:12] in a)
```

`_norm` 只去掉空白字符，不删标点。后半段 `b[:12] in a` 是容错：如果 evidence 有换行或
空格差异，用前 12 个字兜底；如果差异发生在标点上，仍可能匹配失败。它可能误判，
但比严格子串稳。

## 3. `judge` / `judge_ab`

[`eval.py:50`](../../src/babyrag/eval.py#L50)：

```python
result = llm.invoke(JUDGE.format(question=..., gold=..., answer=...),
                    stage="judge", system="你是严格但公正的评测员。")
return parse_score(result.text)
```

`judge_ab`（[`eval.py:57`](../../src/babyrag/eval.py#L57)）用 `JUDGE` 和 `JUDGE_SWAP`
各判一次：

```python
first = llm.invoke(JUDGE.format(...), stage="judge")
second = llm.invoke(JUDGE_SWAP.format(...), stage="judge_swap")
a, b = parse_score(first.text), parse_score(second.text)
return a, b, a == b
```

它只检查位置偏差，不解决“judge 本身判错”的问题。

## 4. `evaluate`：双循环

[`eval.py:69`](../../src/babyrag/eval.py#L69)：

```python
Cache(CACHE_DB).close()
for rung in rungs:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for q in queries:
            mode = "adaptive" if rung == "adaptive" else "ladder"
            futures[pool.submit(_answer_one, settings, q, rung, mode)] = q
        for future in as_completed(futures):
            q = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {... "error": str(exc) ...}
            result["type"] = q.get("type", "")
            result.update(_score(settings, q, result))
            rows.append(result)
```

- 外层 rung，内层题；每个 rung 一个线程池。
- 单题异常不终止整批，写一个 `error` 行。
- 打分在主线程做，`judge` 会用主线程的 `_env`。
- `_score` 看到 `result["error"]` 会立刻返回一整套 `None`，把这条从所有分母里排除；
  `summary` 的 `errors` 单独计数。这不是“答错”，不能按 0 分算。
- `run_name` 参数目前没有被 `evaluate` 使用；输出目录是 `run_eval.py` 自己算好后直接传给
  `write_reports`，不要误以为改这个参数会改变 `per_query.jsonl` 的位置。

## 5. `_score`：单题指标

[`eval.py:116`](../../src/babyrag/eval.py#L116)。

### 5.1 文档级

```python
doc_ids = [set(h.get("doc_ids") or [h.get("doc_id", "")]) for h in hits]
for k in (1, 3, 5, 10):
    out[f"doc_recall@{k}"] = int(any(gold_docs & ids for ids in doc_ids[:k])) if gold_docs else None
doc_rank = next((i for i, ids in enumerate(doc_ids, start=1) if gold_docs & ids), None)
```

`doc_recall` 是“任意一个 gold 文档”；`doc all@5`（[`eval.py:151`](../../src/babyrag/eval.py#L151)）
要求**全部** gold 文档都在前 5。`context_precision@5` 的分母是 `len(hits[:5])`，
不是固定 5；检索只返回 3 条时，它按 3 条算。

### 5.2 Evidence 级

[`eval.py:138`](../../src/babyrag/eval.py#L138)：

```python
def relevant(h):
    if h["chunk_id"].startswith("raptor:"):
        return bool(gold_docs & set(h.get("doc_ids") or []))
    return evidence_in(h.get("text", ""), evidence)

chunk_rank = next((i for i, h in enumerate(hits, start=1) if relevant(h)), None)
```

RAPTOR 摘要按“覆盖 gold 文档”算命中；普通 chunk 按 evidence 子串。

### 5.3 弃答与正确性

[`eval.py:167`](../../src/babyrag/eval.py#L167)：

```python
if should_abstain:
    out["correct"] = None
    out["abstain_correct"] = int(abstain)
else:
    out["abstain_correct"] = None
    if abstain or not result.get("answer"):
        out["correct"] = 0
    else:
        out["correct"] = judge(...)
```

这是分母修正的关键：

- no-answer 题不进 `可答 acc`；
- 所有 rung 都在 no-answer 题上算 `abstain_correct`；
- 空答案/弃答直接 0，不调 judge。

### 5.4 `citation_precision`

[`eval.py:163`](../../src/babyrag/eval.py#L163)：

```python
sum(1 for c in citations if any(c == h.get("chunk_id") for h in hits)) / len(citations)
```

它只检查“引用的 chunk 在不在 hits 里”，不检查是否支持该句。名字叫 precision，实际是覆盖率。

## 6. `summarize`

[`eval.py:188`](../../src/babyrag/eval.py#L188) 按 rung 分组：

```python
correct = [r["correct"] for r in items if r.get("correct") is not None]
entry = {
  "n": len(items),
  "correct_n": len(correct),
  "accuracy": round(sum(correct) / len(correct), 4) if correct else None,
  ...
}
```

- `_mean`（[`eval.py:257`](../../src/babyrag/eval.py#L257)）过滤 `None`；no-answer 题不会拉低
  `chunk@1`。
- `_abstain_accuracy`（[`eval.py:262`](../../src/babyrag/eval.py#L262)）只算 `abstain_correct` 非空。
- `usd_cold_per_query` 按 token 重算（[`eval.py:228`](../../src/babyrag/eval.py#L228)），不受缓存影响；
  `cost["free"]` 为真的行直接记 0，不参与 DeepSeek 主榜的成本比较。

### 综合分

[`eval.py:244`](../../src/babyrag/eval.py#L244)：

```python
quality = (0.50 * (entry["accuracy"] or 0.0)
           + 0.30 * (entry["chunk@1"] or 0.0)
           + 0.20 * (entry["mrr@10"] or 0.0))
factor = 1.0 / (1.0 + 100.0 * (entry["usd_cold_per_query"] or 0.0))
entry["composite"] = round(100.0 * quality * factor, 1)
```

## 7. 置信区间

[`eval.py:267`](../../src/babyrag/eval.py#L267) 是 percentile bootstrap：

```python
rng = random.Random(seed)
means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
return (means[int(0.025*n)], means[int(0.975*n)-1])
```

[`eval.py:276`](../../src/babyrag/eval.py#L276) 是 Wilson：

```python
p = sum(values) / n
z = 1.96
den = 1 + z*z/n
center = (p + z*z/(2*n)) / den
margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
return (max(0, center - margin), min(1, center + margin))
```

报告用 Wilson；bootstrap 留在 `summary.json`。

## 8. `write_reports`

[`eval.py:289`](../../src/babyrag/eval.py#L289)：

- `per_query.jsonl` 去掉 `hits.text`（体积），保留 `chunk_id/doc_ids`。
- `summary.json` 保留全部字段。
- `leaderboard.md` 只展示主要列；`context_precision@5` 在 JSON 里但不进表。

## 调用关系

```text
evaluate
  -> _env (每线程)
  -> run_one (graph)
  -> _score
       -> evidence_in
       -> judge (LLM)
       -> 文档/命中/弃答字段
  -> summarize
       -> _mean / _abstain_accuracy
       -> _bootstrap_ci / _wilson_ci
       -> composite
  -> write_reports
```

## 坑清单

| 坑 | 位置 | 后果 |
|---|---|---|
| 把 no-answer 计进可答 acc | eval.py:167 | 分母不可比 |
| `_mean` 不过滤 None | eval.py:257 | 无 evidence 的题拉低均值 |
| RAPTOR 要求逐字 evidence | eval.py:138 | 摘要永远不命中 |
| `citation_precision` 当“支持度” | eval.py:163 | 高估引用质量 |
| 只看 bootstrap 不报 Wilson | eval.py:267 | 小样本 9/9 会给出 [100,100] |

## 自测

1. 为什么 no-answer 题的 `correct` 是 `None` 而不是 0？
2. RAPTOR 摘要为什么用 `doc_ids` 判断命中？
3. `_mean` 过滤 `None` 会带来什么副作用？

### 答案

1. 它不进可答 acc 的分母；用 0 会污染准确率。弃答能力由 `abstain_correct` 单独衡量。
2. 摘要本来就是抽象，不应逐字包含 evidence；用覆盖的 gold 文档判断更公平。
3. 不同指标的样本数可能不同（例如 chunk@1 只统计有 evidence 的题）；报告里要意识到 `n` 不是总题数。
