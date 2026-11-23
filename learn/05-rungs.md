---
title: Rung 计划表
date: 2026-11-23
input_shape: concept
subject: rungs.py 的九级阶梯与自适应路由
---

# Rung 计划表

`rungs.py` 是整套设计里最短、但最重要的文件：**它把“RAG 发展史”编码成十行计划。**

## RUNGS

[rungs.py:5](../src/babyrag/rungs.py#L5)：

```python
RUNGS = {
    "r0_dense":       ["retrieve_dense", "generate"],
    "r1_hybrid":      ["retrieve_hybrid", "fuse_rrf", "generate"],
    "r2_rerank":      ["retrieve_hybrid", "fuse_rrf", "rerank", "generate"],
    "r3_hyde":        ["hyde", "retrieve_hybrid", "fuse_rrf", "rerank", "generate"],
    "r4_multiquery":  ["rewrite_multi_query", "retrieve_multi", "fuse_rrf", "rerank", "generate"],
    "r5_contextual":  ["retrieve_contextual", "fuse_rrf", "rerank", "generate"],
    "r6_raptor":      ["retrieve_raptor", "rerank", "generate"],
    "r7_crag":        ["retrieve_hybrid", "fuse_rrf", "rerank", "grade", "generate"],
    "r8_agentic":     ["agentic", "verify", "generate"],
}
```

每个字符串必须是 `graph.STAGES` 里的节点名（[graph.py:32](../src/babyrag/graph.py#L32)）。
`build_graph` 会把 `STAGES` 全部注册成节点，所以计划表只需写名字。

## 这是累积阶梯，不是单变量消融

字面上每个 rung“只多一个节点”，但**状态是累积的**：

- `r3_hyde` 同时拥有 hybrid、RRF、rerank；它的差值包含前面所有改动。
- `r4_multiquery` 同样叠加了 hybrid + rerank。
- `r5_contextual` 没有 `retrieve_hybrid`，因为它换成 contextual 检索器，但仍要 RRF + rerank。

所以排行榜里“r2 比 r1 高”可以解释为 rerank 的贡献，“r3 比 r2 低”只能解释为
**在 r2 基础上加 HyDE 的净效果**，不能单独说 HyDE 本身如何。要严格归因，需要另建单因素实验。

## adaptive_plan

[rungs.py:47](../src/babyrag/rungs.py#L47)：

```python
def adaptive_plan(question, qtype=None):
    if qtype == "multi":  return "r8_agentic", "type=multi"
    if qtype in {"none", "no_answer"}: return "r7_crag", "type=no_answer"
    if qtype == "global": return "r6_raptor", "type=global"
    if len(question) > 60: return "r4_multiquery", "long_question"
    return "r1_hybrid", "default"
```

它是**纯规则**，不调用 LLM。注意 docstring 写着 “LLM is only a fallback in the node”，
但 `Runner.plan`（[graph.py:58](../src/babyrag/graph.py#L58)）在 adaptive 模式下只调用这个函数，
并没有 LLM 兜底；`ADAPTIVE` 提示词（[prompts.py:54](../src/babyrag/prompts.py#L54)）目前也没被使用。
这是代码与注释的轻微不一致，行为以代码为准。

## LABELS 与 PAPERS

[rungs.py:20](../src/babyrag/rungs.py#L20) 给报告用的短标签；
[rungs.py:33](../src/babyrag/rungs.py#L33) 给论文锚点。`babyrag info` 和报告都读它们，
新增 rung 时记得两处都加，否则表格会显示空。

## 新增一个 rung

1. 在 `RUNGS` 加一行，写节点名。
2. 如果用了新节点，在 `graph.STAGES` 注册，并在 `Runner` 里实现同名方法。
3. 在 `LABELS` / `PAPERS` 加说明。
4. `ALL_RUNGS` 会自动包含它（[rungs.py:18](../src/babyrag/rungs.py#L18)）。
5. 不需要改 `build_graph` 的拓扑——这是这套设计的核心卖点。

## Check yourself

1. 为什么 `r5_contextual` 的计划里没有 `retrieve_hybrid`？
2. `r7_crag` 的 `grade` 在 `generate` 之前，它可能改变什么？
3. 新增 rung 时，哪两个字典最容易漏？

### Answers

1. 它需要走 contextual 的向量/BM25 集合，`retrieve_hybrid` 用的是基础集合；混用会测不到 contextual 的效果。
2. 它会过滤不相关的 chunk，甚至在 `incorrect` 时改写查询并重新检索一次；进入 `generate` 的 hits 可能已经变了。
3. `LABELS` 和 `PAPERS`。`RUNGS` 漏了会 KeyError；后两个漏了只是报告缺文字。
