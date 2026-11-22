# baby-rag-history · masked + hard (25 fair)

同一语料、同一批问题；`N/A` 表示该档不具备该能力，不是 0 分。

| rung | 里程碑 | acc | doc@1 | chunk@1 | chunk@3 | chunk@5 | MRR@10 | nDCG@10 | abstain | cold $/q | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r0_dense` | Naive RAG (dense top-5) | 92.0% | 88.0% | 64.0% | 84.0% | 88.0% | 0.7347 | 0.7712 | N/A | $0.00035 | 3165ms | 4585ms |
| `r1_hybrid` | Hybrid BM25 + dense + RRF | 92.0% | 92.0% | 64.0% | 80.0% | 88.0% | 0.7333 | 0.7702 | N/A | $0.00035 | 2894ms | 46647ms |
| `r2_rerank` | Hybrid + bge-reranker | 92.0% | 96.0% | 80.0% | 84.0% | 84.0% | 0.8133 | 0.82 | N/A | $0.00036 | 3998ms | 39400ms |
| `r3_hyde` | HyDE hypothetical document | 84.0% | 88.0% | 76.0% | 80.0% | 80.0% | 0.78 | 0.7852 | N/A | $0.00095 | 649ms | 34937ms |
| `r4_multiquery` | Multi-query / RAG-Fusion | 96.0% | 96.0% | 80.0% | 84.0% | 92.0% | 0.84 | 0.8597 | N/A | $0.00055 | 3688ms | 34718ms |
| `r5_contextual` | Contextual Retrieval | 92.0% | 96.0% | 80.0% | 84.0% | 84.0% | 0.8133 | 0.82 | N/A | $0.00034 | 3357ms | 30661ms |
| `r6_raptor` | RAPTOR summary tree | 92.0% | 96.0% | 80.0% | 84.0% | 84.0% | 0.8133 | 0.82 | N/A | $0.00035 | 617ms | 4009ms |
| `r7_crag` | CRAG grade + abstain | 92.0% | 96.0% | 80.0% | 84.0% | 84.0% | 0.8133 | 0.82 | N/A | $0.00083 | 8275ms | 43992ms |
| `r8_agentic` | Agentic ReAct | 92.0% | 92.0% | 64.0% | 80.0% | 88.0% | 0.7333 | 0.7702 | N/A | $0.00112 | 2789ms | 35193ms |
| `adaptive` | Adaptive router | 92.0% | 92.0% | 64.0% | 80.0% | 88.0% | 0.7333 | 0.7702 | N/A | $0.00035 | 1439ms | 25541ms |

## 说明

- 25 道公平手写难题 × 去标题 + 干扰文档语料（1,200 篇 / 5,676 chunks）。
- `cold $/q` 按本次记录的 token × 单价计算，不受缓存影响；`本次实付` 是实际账单。
- `chunk@1` / `MRR@10` 是主指标；`doc R@5` 在原始语料上会饱和。
- 成本逐条见 `results/cost.jsonl`。
