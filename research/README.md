# 研究笔记（实现前）

> **免责声明**：这些文档是 2026-11-20 实现前的调研，假设是 **200 篇 / 869 chunks / 30 题 / $10 预算**。
> 最终实现扩到了 1,000 篇，题库也变成了 30 原题 + 25 难题 + 10 test + 25 公开题。
> **现状数字以 [`../docs/results.md`](../docs/results.md) 和 `results/*/summary.json` 为准**，
> 本目录只用于追溯当时的决策依据。

| 文件 | 内容 | 已过期的部分 |
|---|---|---|
| [`milestones.md`](milestones.md) | 17 个候选里程碑的可行性、成本、取舍 | 成本按 869 chunks 估算 |
| [`cost-model.md`](cost-model.md) | 逐 token 成本模型与可复算脚本说明 | 200 篇 / 150 题口径 |
| [`cost_model.py`](cost_model.py) | 成本模型的可复算脚本 | 同上 |
| [`model.json`](model.json) | 成本模型输出 | 同上 |
| [`langgraph-design.md`](langgraph-design.md) | LangGraph 单图多里程碑设计、缓存、评测 | 部分 API/参数是当时版本 |
| [`corpus-eval.md`](corpus-eval.md) | 语料、题库、天花板效应分析 | 当时只有 30 题 |
| [`prior-art.md`](prior-art.md) | RAG_Techniques / rag-from-scratch / LangGraph 教程对照 | 星数与链接是当时快照 |

## 仍然有效的结论

- **天花板效应**：原题 23/30 含 gold 标题，`doc R@5` 会饱和。
- **GraphRAG 是 ROI 决策，不是预算决策**：买得起，但 1,000 篇上社区太薄。
- **缓存是成本关键**：contextual / HyDE / multi-query / grade / judge 都要缓存。
- **评测要报 chunk 级召回**：`doc R@5` 不够。

## 已经被实现推翻/修正的结论

- 「Agentic 是子图」：实现里是带循环的普通节点。
- 「200 篇足够」：实际扩到 1,000 篇并加了 masked 变体。
- 「30 题够用」：实际加了 25 道难题、10 道 test、25 道 RAGBench。
- 「GraphRAG $1.11」：那是 200 篇口径；1,000 篇约 $3.5–6。
