# baby-rag-history · leaderboard

同一语料、同一批问题；`N/A` 表示该档不具备该能力，不是 0 分。

| rung | 里程碑 | acc | doc@1 | chunk@1 | chunk@3 | chunk@5 | MRR@10 | nDCG@10 | abstain | cold $/q | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r0_dense` | Naive RAG (dense top-5) | 88.9% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | $0.00030 | 130ms | 7366ms |
| `r1_hybrid` | Hybrid BM25 + dense + RRF | 88.9% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | $0.00030 | 185ms | 19232ms |
| `r2_rerank` | Hybrid + bge-reranker | 96.3% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | $0.00030 | 565ms | 18701ms |
| `r3_hyde` | HyDE hypothetical document | 96.3% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | $0.00081 | 651ms | 19918ms |
| `r4_multiquery` | Multi-query / RAG-Fusion | 92.6% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | $0.00046 | 1094ms | 20102ms |
| `r5_contextual` | Contextual Retrieval | 100.0% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | $0.00031 | 474ms | 19854ms |
| `r6_raptor` | RAPTOR summary tree | 96.3% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | $0.00030 | 535ms | 752ms |
| `r7_crag` | CRAG grade + abstain | 93.3% | N/A | N/A | N/A | N/A | N/A | N/A | 100.0% | $0.00065 | 610ms | 15314ms |
| `r8_agentic` | Agentic ReAct | 90.0% | N/A | N/A | N/A | N/A | N/A | N/A | 100.0% | $0.00086 | 378ms | 19572ms |
| `adaptive` | Adaptive router | 90.0% | N/A | N/A | N/A | N/A | N/A | N/A | 100.0% | $0.00045 | 956ms | 19527ms |

## 说明

- 30 道原题；1,000 篇 / 4,740 chunks。
- `cold $/q` 按本次记录的 token × 单价计算，不受缓存影响；`本次实付` 是实际账单。
- `chunk@1` / `MRR@10` 是主指标；`doc R@5` 在原始语料上会饱和。
- 成本逐条见 `results/cost.jsonl`。
