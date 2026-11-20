# baby-rag-history · leaderboard

同一语料、同一批问题；`N/A` 表示该档不具备该能力，不是 0 分。

| rung | 里程碑 | acc | doc R@5 | chunk hit@5 | abstain | cold $/q | 本次实付 | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r0_dense` | Naive RAG (dense top-5) | 88.9% | 100.0% | 96.3% | N/A | $0.00030 | $0.001 | 130ms | 7366ms |
| `r1_hybrid` | Hybrid BM25 + dense + RRF | 88.9% | 100.0% | 96.3% | N/A | $0.00030 | $0.000 | 185ms | 19232ms |
| `r2_rerank` | Hybrid + bge-reranker | 96.3% | 100.0% | 96.3% | N/A | $0.00030 | $0.000 | 565ms | 18701ms |
| `r3_hyde` | HyDE hypothetical document | 96.3% | 100.0% | 96.3% | N/A | $0.00081 | $0.001 | 651ms | 19918ms |
| `r4_multiquery` | Multi-query / RAG-Fusion | 92.6% | 100.0% | 96.3% | N/A | $0.00046 | $0.001 | 1094ms | 20102ms |
| `r5_contextual` | Contextual Retrieval | 100.0% | 100.0% | 100.0% | N/A | $0.00031 | $0.000 | 474ms | 19854ms |
| `r6_raptor` | RAPTOR summary tree | 96.3% | 100.0% | 96.3% | N/A | $0.00030 | $0.000 | 535ms | 752ms |
| `r7_crag` | CRAG grade + abstain | 93.3% | 100.0% | 92.6% | 100.0% | $0.00065 | $0.001 | 610ms | 15314ms |
| `r8_agentic` | Agentic ReAct | 90.0% | 100.0% | 96.3% | 100.0% | $0.00086 | $0.000 | 378ms | 19572ms |
| `adaptive` | Adaptive router | 90.0% | 100.0% | 96.3% | 100.0% | $0.00045 | $0.000 | 956ms | 19527ms |

## 说明

- 评测集 30 题（single 12 / entity 6 / multi 6 / global 3 / none 3）。
- global 与 none 各 3 题，只适合演示方向，不能下统计结论。
- `cold $/q` 按本次记录的 token × 单价计算，不受缓存影响；`本次实付` 是实际账单。
- 语料 1,000 篇 / 4,740 chunks；top-5 的 doc 召回已接近饱和，差异主要来自 chunk 级排序。
- 成本逐条见 `results/cost.jsonl`。
