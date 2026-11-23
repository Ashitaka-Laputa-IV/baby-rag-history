---
title: 综合练习：改一个 rung、加一个指标
date: 2026-11-23
input_shape: concept
subject: baby-rag-history 源码改造练习
---

# 综合练习

三个由浅入深的改造。都只需要读 `src/`，不需要重跑索引。

## 练习 A：只改计划表，加一个 rung（20 分钟）

目标：新增 `r9_hybrid_top10`，计划是 `retrieve_hybrid → fuse_rrf → generate`，但让
`retrieve_hybrid` 召回 20 条。

提示：`RUNGS` 只是计划表，节点参数目前写死在 `_hybrid`（[graph.py:52](../src/babyrag/graph.py#L52)）
里。你会发现**只加一行 `RUNGS` 不会改变召回数**——这正是“计划表只控制顺序，不控制参数”的边界。

验收：

- `babyrag info` 能看到新 rung；
- `babyrag ask ... --rung r9_hybrid_top10` 能跑；
- 说明为什么它和 `r1_hybrid` 的结果相同。

## 练习 B：把 top-k 变成配置（1–2 小时）

目标：新增 `FINAL_TOP_K` 和 `RETRIEVE_K` 到 [config.py:71](../src/babyrag/config.py#L71)，
让 `generate`、`rerank`、`_hybrid` 都读它；然后加一个 `r9_top3` rung，验证 top-3 对
`evidence R@1` 和 `可答 acc` 的影响。

需要改：

| 文件 | 位置 | 改什么 |
|---|---|---|
| `config.py` | [71](../src/babyrag/config.py#L71) | 加常量 |
| `graph.py` | [52](../src/babyrag/graph.py#L52) | `_hybrid` 的 `k` |
| `graph.py` | [118](../src/babyrag/graph.py#L118) | `rerank(... top_n=...)` |
| `graph.py` | [232](../src/babyrag/graph.py#L232) | `hits[:5]` |
| `retrieval.py` | [210](../src/babyrag/retrieval.py#L210) | `build_context` 的 `budget` 不变，但调用方传 top-k |
| `eval.py` | [289](../src/babyrag/eval.py#L289) | 报告表头里的 “top-5” 说明 |

验收：

- `r9_top3` 与 `r2_rerank` 只差最终条数；
- `results/*/summary.json` 里 `evidence R@1` 变化合理（top-3 更容易让 gold 排第一，但上下文更少）；
- 不破坏 `r0`–`r8` 的旧结果（用 `git diff` 检查默认值没变）。

## 练习 C：加一个引用指标（1 小时）

目标：在 `_score` 里加 `citation_gold`：答案引用的 chunk 中，至少一个属于 gold 文档。

提示：`result["citations"]` 是 chunk_id 列表；`result["hits"]` 有 `doc_ids`。
需要：

1. 建 `chunk_id → doc_ids` 映射；
2. 判断 cited 是否与 `gold_docs` 相交；
3. 在 `summarize` 加 `_mean`，在 `write_reports` 加一列；
4. 在 `docs/evaluation.md` 补定义。

验收：

- `results/*/summary.json` 有 `citation_gold`；
- 报告表和 README 的指标说明一致；
- 至少一个 rung 的值不是 100%（否则说明指标写错了）。

## 练习 D（可选）：修 masked 的标题泄漏

读 [build_masked_corpus.py](../scripts/build_masked_corpus.py) 和
[04-retrieval.md](04-retrieval.md) 的说明：masked 只替换了 chunk 文本，metadata 的 `title`
仍会被 `build_context` 拼回上下文。

任务：

1. 在 `build_masked_corpus.py` 里把 masked chunk 的 `title` 也设为「（标题已隐藏）」；
2. 重建 `chunks.masked.jsonl` 和 `chunks_masked` 集合（embedding 免费）；
3. 重跑 masked 评测；
4. 比较新旧 `可答 acc` / `evidence R@1`，写进 `docs/results.md` 的“已知偏差”。

验收：报告明确写“完全去标题”还是“部分去偏”，不再含糊。

## Check yourself

1. 练习 A 为什么不会改变结果？
2. 练习 B 里哪些地方必须一起改，漏一个会怎样？
3. 练习 C 的 `citation_gold` 和现有 `citation_precision` 有什么区别？

### Answers

1. `RUNGS` 只决定节点顺序；`_hybrid` 的 `k=10`、`generate` 的 `[:5]` 是写死的，计划表改不到它们。
2. `_hybrid`、`rerank`、`generate` 三处必须一起改；只改 `generate` 会让 rerank 仍返回 5 条，只改 `_hybrid` 会让上游多召回、下游仍截 5。
3. `citation_precision` 只检查“引用的 chunk 在不在 hits 里”；`citation_gold` 检查“引用的 chunk 是否属于 gold 文档”，后者才和证据相关。
