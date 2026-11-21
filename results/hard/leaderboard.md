# baby-rag-history · hard questions

同一语料、同一批问题；`N/A` 表示该档不具备该能力，不是 0 分。

| rung | 里程碑 | acc | doc@1 | chunk@1 | chunk@3 | chunk@5 | MRR@10 | nDCG@10 | abstain | cold $/q | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r0_dense` | Naive RAG (dense top-5) | 79.3% | 79.3% | 62.1% | 82.8% | 86.2% | 0.7213 | 0.757 | N/A | $0.00037 | 3481ms | 33949ms |
| `r1_hybrid` | Hybrid BM25 + dense + RRF | 79.3% | 79.3% | 55.2% | 79.3% | 86.2% | 0.6782 | 0.7247 | N/A | $0.00038 | 3608ms | 35399ms |
| `r2_rerank` | Hybrid + bge-reranker | 79.3% | 89.7% | 72.4% | 86.2% | 89.7% | 0.7943 | 0.82 | N/A | $0.00037 | 3779ms | 32188ms |
| `r3_hyde` | HyDE hypothetical document | 75.9% | 79.3% | 65.5% | 79.3% | 86.2% | 0.7264 | 0.7598 | N/A | $0.00096 | 13603ms | 43782ms |
| `r4_multiquery` | Multi-query / RAG-Fusion | 82.8% | 89.7% | 75.9% | 89.7% | 89.7% | 0.8218 | 0.8411 | N/A | $0.00055 | 7818ms | 54874ms |
| `r5_contextual` | Contextual Retrieval | 82.8% | 89.7% | 75.9% | 86.2% | 89.7% | 0.8115 | 0.8327 | N/A | $0.00036 | 4014ms | 41830ms |
| `r6_raptor` | RAPTOR summary tree | 79.3% | 89.7% | 72.4% | 86.2% | 89.7% | 0.7943 | 0.82 | N/A | $0.00036 | 609ms | 4938ms |
| `r7_crag` | CRAG grade + abstain | 79.3% | 89.7% | 75.9% | 86.2% | 89.7% | 0.8115 | 0.8327 | N/A | $0.00085 | 10986ms | 45644ms |
| `r8_agentic` | Agentic ReAct | 79.3% | 79.3% | 55.2% | 79.3% | 86.2% | 0.6782 | 0.7247 | N/A | $0.00104 | 15116ms | 48687ms |
| `adaptive` | Adaptive router | 79.3% | 79.3% | 55.2% | 79.3% | 86.2% | 0.6782 | 0.7247 | N/A | $0.00038 | 233ms | 30223ms |

## 说明

- 29 道手写难题：避开文档标题，保留年份/地区/职业等限定词；evidence 已校验落在 gold 文档内。
- `cold $/q` 按本次记录的 token × 单价计算，不受缓存影响；`本次实付` 是实际账单。
- `chunk@1` / `MRR@10` 是主指标；`doc R@5` 在原始语料上会饱和。
- 成本逐条见 `results/cost.jsonl`。
