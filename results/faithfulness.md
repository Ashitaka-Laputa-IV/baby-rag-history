# Faithfulness / citation support

Sampled 30 non-abstaining answers (max 3 per rung).

| rung | n | supported | citation_ok | unsupported claims |
|---|---:|---:|---:|---:|
| `adaptive` | 3 | 1.00 | 100% | 0 |
| `r0_dense` | 3 | 1.00 | 100% | 1 |
| `r1_hybrid` | 3 | 1.00 | 100% | 0 |
| `r2_rerank` | 3 | 1.00 | 100% | 0 |
| `r3_hyde` | 3 | 1.00 | 100% | 0 |
| `r4_multiquery` | 3 | 1.00 | 100% | 0 |
| `r5_contextual` | 3 | 1.00 | 100% | 0 |
| `r6_raptor` | 3 | 1.00 | 100% | 0 |
| `r7_crag` | 3 | 1.00 | 100% | 0 |
| `r8_agentic` | 3 | 1.00 | 100% | 0 |

`supported` is the judge's estimated fraction of factual claims backed by the numbered context; `citation_ok` is whether citations point at supporting passages.
One answer failed JSON parsing and is excluded from the averages (still counted in n).
