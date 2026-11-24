# 文档索引

按阅读目的选择：

| 文档 | 内容 | 适合谁 |
|---|---|---|
| [`../learn/README.md`](../learn/README.md) | 按调用链读源码的通俗教程 | 第一次读代码的人 |
| [`../learn/code-walkthrough/README.md`](../learn/code-walkthrough/README.md) | 逐函数代码漫游，贴真实实现 | 要 review / 改代码的人 |
| [项目网页](https://ashitaka-laputa-iv.github.io/baby-rag-history/) | 浅色单页：九个台阶、结果、架构图 | 想快速看全貌的人 |
| [`../README.md`](../README.md) | 项目是什么、怎么跑、核心结论 | 第一次看的人 |
| [`design.md`](design.md) | 架构、rung 设计、数据管线、关键决策 | 要改代码的人 |
| [`results.md`](results.md) | 六套评测的完整结果与解读 | 要看数字的人 |
| [`evaluation.md`](evaluation.md) | 指标定义、评测协议、偏差与限制 | 要判断结论可信度的人 |
| [`history.md`](history.md) | 九个 RAG 里程碑的论文与机制 | 要补背景的人 |
| [`graphrag-not-built.md`](graphrag-not-built.md) | 为什么不做 GraphRAG | 关心取舍的人 |
| [`../research/README.md`](../research/README.md) | 实现前的研究笔记（含过期假设） | 要追溯决策的人 |

## 推荐阅读顺序

1. `README.md`：先看结论。
2. `results.md`：看六套数字和「谁赢在哪」。
3. `evaluation.md`：确认这些数字是怎么算的、哪里不可信。
4. `design.md`：再看实现。

## 数字的唯一来源

所有结果数字以 `results/*/summary.json` 和 `results/*/leaderboard.md` 为准。
`research/` 下的文档是**实现前**的研究，里面的 200 篇 / 30 题 / 成本估算可能已经过期，不要当现状引用。
