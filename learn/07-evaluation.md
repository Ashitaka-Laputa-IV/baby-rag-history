---
title: 评测
date: 2026-11-23
input_shape: concept
subject: eval.py 的指标、分母与报告
---

# 评测

`eval.py` 是“跑批量 + 算指标 + 写报告”。它不实现 RAG，只调用 `graph` 和 `judge`。

## 批量入口

[evaluate:69](../src/babyrag/eval.py#L69)：

```python
Cache(CACHE_DB).close()          # 先在主线程建表，避免工作线程竞争
for rung in rungs:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for q in queries:
            futures[pool.submit(_answer_one, settings, q, rung, mode)] = q
```

- 每个 rung 一个线程池，不是所有 rung 混跑；这样进度输出按 rung 分组。
- `_env`（[eval.py:24](../src/babyrag/eval.py#L24)）用 `threading.local()` 给每个线程建
  `Services + LLMClient + graph`，避免 Chroma/SQLite 跨线程共享。
- 单题异常不会中断整批：`future.result()` 抛错时写一个 `error` 行（[eval.py:91](../src/babyrag/eval.py#L91)）。

## 一题怎么打分

[_score:116](../src/babyrag/eval.py#L116) 输入是 `result`（`run_one` 的输出）和 `q`（评测题）。

### 文档级

```python
doc_ids = [set(h.get("doc_ids") or [h.get("doc_id", "")]) for h in hits]
for k in (1, 3, 5, 10):
    doc_recall@k = any(gold_docs & ids for ids in doc_ids[:k])
```

`doc all@5`（[eval.py:137](../src/babyrag/eval.py#L137)）更严格：要求**所有** gold 文档都在前 5 条，
单跳题与 `doc_recall@5` 相同，多跳题才有区别。

### Evidence 级

[eval.py:127](../src/babyrag/eval.py#L127)：

```python
def relevant(h):
    if h["chunk_id"].startswith("raptor:"):
        return bool(gold_docs & set(h.get("doc_ids") or []))
    return evidence_in(h.get("text", ""), evidence)
```

- 普通 chunk：`evidence` 子串出现在文本里才算命中（[eval.py:42](../src/babyrag/eval.py#L42)）。
- RAPTOR 摘要：不要求逐字包含 evidence，只要摘要覆盖 gold 文档。这是为了避免对摘要节点不公平。

然后：

| 字段 | 含义 |
|---|---|
| `chunk@1/3/5` | gold evidence 第一次出现在第几名（1/3/5 内） |
| `mrr@10` | `1/rank`，没命中为 0 |
| `ndcg@10` | `1/log2(rank+1)`，二值相关 |

### 弃答与正确性

[eval.py:156](../src/babyrag/eval.py#L156)：

```python
if should_abstain:
    out["correct"] = None
    out["abstain_correct"] = int(abstain)
else:
    out["abstain_correct"] = None
    out["correct"] = judge(...)  # 或 0（弃答/空答案）
```

这是 v2 的关键修正：

- **no-answer 题永远不进可答 acc 的分母**；
- 所有 rung 都在 no-answer 题上算 `abstain_correct`，不会因为没有弃答路径就“跳过”。

`judge`（[eval.py:50](../src/babyrag/eval.py#L50)）调 LLM；`judge_ab`（[eval.py:57](../src/babyrag/eval.py#L57)）
用 `JUDGE` + `JUDGE_SWAP` 各判一次，检查位置偏差。

## 汇总

[summarize:177](../src/babyrag/eval.py#L177) 按 rung 分组：

- `_mean`（[eval.py:243](../src/babyrag/eval.py#L243)）**过滤 None**；所以 no-answer 题不会把
  `chunk@1` 拉低。
- `accuracy` 只算 `correct is not None` 的题；`correct_n` 是可答题数。
- `abstain_accuracy`（[eval.py:248](../src/babyrag/eval.py#L248)）只算 `abstain_correct` 非空。
- `usd_cold_per_query` 按 token 重算，不受缓存影响（[eval.py:211](../src/babyrag/eval.py#L211)）。
- Wilson CI（[eval.py:262](../src/babyrag/eval.py#L262)）用于报告；bootstrap（[eval.py:253](../src/babyrag/eval.py#L253)）
  仍写进 `summary.json`。

### 综合分

[eval.py:230](../src/babyrag/eval.py#L230)：

```python
quality = 0.50 * accuracy + 0.30 * chunk@1 + 0.20 * mrr@10
factor = 1 / (1 + 100 * usd_cold_per_query)
composite = 100 * quality * factor
```

- 成本项是温和惩罚：$0.001/q 约 9% 折扣，$0.0003/q 约 3%。
- 弃答没有折进综合分，单独一列。

## 报告

[write_reports:275](../src/babyrag/eval.py#L275)：

- `per_query.jsonl` 去掉 `hits.text` 以控体积，但保留 `chunk_id/doc_ids`，方便复算。
- `summary.json` 保留所有字段（包括 bootstrap CI、context_precision）。
- `leaderboard.md` 只展示主要列；`context_precision@5` 仍在 JSON 里但**不进表**，
  因为单 evidence 题最大只有 1/5，跨 rung 不可比。

## 一个容易忽略的点

`_score` 里 `chunk@1` 对没有 `evidence` 的题返回 `None`，`_mean` 会过滤；但 `doc@1` 对没有
`gold_docs` 的题返回 `None`，同样被过滤。写新指标时要保持一致，否则分母会变。

## Check yourself

1. 为什么 r0–r6 的 `abstain_correct` 也要算，而不是标 N/A？
2. 为什么 RAPTOR 的摘要节点不按 evidence 子串算命中？
3. `context_precision@5` 为什么不出现在 leaderboard？

### Answers

1. 弃答是能力维度，所有 rung 都应该被同一把尺子量；不能弃答的 rung 得 0 是诚实结果，标 N/A 会掩盖差异。
2. 摘要本来就是抽象，不应逐字包含原文；按“覆盖 gold 文档”算命中才公平，否则 RAPTOR 永远在 chunk 指标上吃亏。
3. 单 evidence 题最多只有 1/5 相关，跨 rung/题型的值不可比；CRAG 的高分来自“只保留少量 chunk”，不是检索更好。保留在 JSON 供审计即可。
