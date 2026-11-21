# baby-rag-history · masked + hard questions

同一语料、同一批问题；`N/A` 表示该档不具备该能力，不是 0 分。

| rung | 里程碑 | acc | doc@1 | chunk@1 | chunk@3 | chunk@5 | MRR@10 | nDCG@10 | abstain | cold $/q | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r0_dense` | Naive RAG (dense top-5) | 79.3% | 79.3% | 58.6% | 75.9% | 79.3% | 0.6678 | 0.6993 | N/A | $0.00038 | 3224ms | 4856ms |
| `r1_hybrid` | Hybrid BM25 + dense + RRF | 79.3% | 79.3% | 55.2% | 69.0% | 79.3% | 0.6408 | 0.6788 | N/A | $0.00038 | 3050ms | 46647ms |
| `r2_rerank` | Hybrid + bge-reranker | 79.3% | 89.7% | 72.4% | 75.9% | 75.9% | 0.7356 | 0.7414 | N/A | $0.00038 | 4147ms | 39400ms |
| `r3_hyde` | HyDE hypothetical document | 72.4% | 82.8% | 69.0% | 72.4% | 72.4% | 0.7069 | 0.7114 | N/A | $0.00097 | 600ms | 34937ms |
| `r4_multiquery` | Multi-query / RAG-Fusion | 82.8% | 89.7% | 72.4% | 75.9% | 82.8% | 0.7586 | 0.7756 | N/A | $0.00056 | 3852ms | 34718ms |
| `r5_contextual` | Contextual Retrieval | 79.3% | 89.7% | 72.4% | 75.9% | 75.9% | 0.7356 | 0.7414 | N/A | $0.00037 | 3466ms | 30661ms |
| `r6_raptor` | RAPTOR summary tree | 79.3% | 89.7% | 72.4% | 75.9% | 75.9% | 0.7356 | 0.7414 | N/A | $0.00037 | 606ms | 4420ms |
| `r7_crag` | CRAG grade + abstain | 79.3% | 89.7% | 72.4% | 75.9% | 75.9% | 0.7356 | 0.7414 | N/A | $0.00090 | 10714ms | 43992ms |
| `r8_agentic` | Agentic ReAct | 79.3% | 79.3% | 55.2% | 69.0% | 79.3% | 0.6408 | 0.6788 | N/A | $0.00115 | 2789ms | 35193ms |
| `adaptive` | Adaptive router | 79.3% | 79.3% | 55.2% | 69.0% | 79.3% | 0.6408 | 0.6788 | N/A | $0.00038 | 1430ms | 25541ms |

## 说明

- 29 道手写难题 × 去标题 + 干扰文档语料（1,200 篇 / 5,676 chunks）。
- `cold $/q` 按本次记录的 token × 单价计算，不受缓存影响；`本次实付` 是实际账单。
- `chunk@1` / `MRR@10` 是主指标；`doc R@5` 在原始语料上会饱和。
- 成本逐条见 `results/cost.jsonl`。
