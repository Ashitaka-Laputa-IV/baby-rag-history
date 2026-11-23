# 设计

> 结果见 [`results.md`](results.md)；指标与偏差见 [`evaluation.md`](evaluation.md)。

## 1. 目标与非目标

**目标**：用一份中文维基语料，把 RAG 发展史上的九个里程碑放进**同一张 LangGraph 状态机**，
逐级只改一个变量，并同时记录效果、召回、弃答、延迟和成本。

**非目标**：
- 不是论文复现。DPR 训练、Self-RAG reflection token、CRAG 网页回退都用同机制的 LLM/规则替代。
- 不追求排行榜 SOTA；追求「每个 rung 多了什么、贵多少、还有什么做不到」。
- 不做 GraphRAG（理由见 [`graphrag-not-built.md`](graphrag-not-built.md)）。

## 2. 架构：一张图 + 计划表

```text
START → plan → dispatch ─┬→ retrieve_dense / retrieve_hybrid / retrieve_contextual
                         ├→ retrieve_multi / retrieve_raptor
                         ├→ hyde / rewrite_multi_query / fuse_rrf / rerank
                         ├→ grade / agentic / verify / generate
                         └→ finalize → END
```

- `plan` 只产出 `plan: list[str]` 和 `cursor`；`dispatch` 是唯一路由点，读 `plan[cursor]`。
- 每个阶段节点只做一件事，做完 `cursor + 1` 回 `dispatch`；越界进 `finalize`。
- **rung 是一行计划，不是一张图**。新增一个里程碑 = 加节点 + 表里加一行，不改拓扑。
- **`agentic` 是带 4 步循环的普通节点**，不是独立的 LangGraph 子图。
- 最终上下文固定 **top-5 / 4,000 token**，所有 rung 同一预算。

代码位置：

| 文件 | 职责 |
|---|---|
| [`src/babyrag/rungs.py`](../src/babyrag/rungs.py) | `RUNGS` 计划表、adaptive 规则、论文锚点 |
| [`src/babyrag/graph.py`](../src/babyrag/graph.py) | 节点、`build_graph`、`run_one` |
| [`src/babyrag/retrieval.py`](../src/babyrag/retrieval.py) | dense / BM25 / RRF / rerank / context |
| [`src/babyrag/eval.py`](../src/babyrag/eval.py) | 指标、汇总、报告 |
| [`src/babyrag/llm.py`](../src/babyrag/llm.py) | ChatOpenAI、embedding、rerank、成本与重试 |
| [`src/babyrag/cache.py`](../src/babyrag/cache.py) | SQLite 缓存（WAL + busy timeout） |

## 3. Rung 计划

| rung | 计划 | 改的变量 |
|---|---|---|
| `r0_dense` | `retrieve_dense → generate` | 基线 |
| `r1_hybrid` | `retrieve_hybrid → fuse_rrf → generate` | 加 BM25 |
| `r2_rerank` | `retrieve_hybrid → fuse_rrf → rerank → generate` | 加精排 |
| `r3_hyde` | `hyde → retrieve_hybrid → fuse_rrf → rerank → generate` | 查询表示 |
| `r4_multiquery` | `rewrite_multi_query → retrieve_multi → fuse_rrf → rerank → generate` | 查询数量 |
| `r5_contextual` | `retrieve_contextual → fuse_rrf → rerank → generate` | 索引侧语境 |
| `r6_raptor` | `retrieve_raptor → rerank → generate` | 摘要树 |
| `r7_crag` | `retrieve_hybrid → fuse_rrf → rerank → grade → generate` | 评估 + 弃答 |
| `r8_agentic` | `agentic → verify → generate` | 多轮检索 |
| `adaptive` | 规则路由到上面一档 | 何时用哪档 |

**注意**：这是**累积阶梯**，不是单变量消融。例如 r3 同时拥有 hybrid、rerank、HyDE；
rung 之间的差值包含前面所有改动。严格的单因素消融需要另建实验。

## 4. 数据管线

### 4.1 主语料

1. 来源：`wikimedia/wikipedia` 20231101 中文 shard（`zhwiki-00002.parquet`）。
2. 过滤：800–4,000 字、CJK 比例 ≥ 0.3、排除列表/消歧义/车站等标题。
3. 主题：LLM 分 person/geo/tech/hist/art 五类；失败回退规则分类。
4. 抽样：`rank = blake2b(seed:title)`，按主题交错排序，嵌套抽样。
5. 切分：chonkie 512 token / overlap 64；超长句先用 `fix_long_sentences.py` 预处理。
6. 规模：**1,000 篇 / 五主题各 200 / 4,740 chunks**。

### 4.2 masked 变体

- chunk 标题替换为「（标题已隐藏）」；
- 另加 **200 篇同标题前缀**的干扰文档；
- 规模：**1,200 篇 / 5,676 chunks**。
- 目的：原题 30 道里有 23 道直接包含 gold 标题，不 mask 就无法测出检索差异。

### 4.3 索引

| 索引 | 内容 | 成本 |
|---|---|---|
| dense | bge-m3 + Chroma cosine | 免费 |
| BM25 | jieba + rank_bm25（内存） | 免费 |
| contextual | 每个 chunk 的定位前缀 + 独立向量/BM25 | 4,740 次 LLM，约 $1.53 |
| RAPTOR | 527 个一级 + 14 个二级摘要 | 约 541 次 LLM，约 $0.34 |
| masked contextual | 复用原前缀、替换标题、重新嵌入 | 约 $0.45（近似） |

### 4.4 问题集

| 集合 | 题数 | 用途 |
|---|---:|---|
| `eval/queries.jsonl` | 30（single 12 / entity 6 / multi 6 / global 3 / none 3） | 主榜 |
| `eval/queries-hard.jsonl` | 29 道手写，剔除 4 道指代不清后 **25 道公平题** | 去标题线索 |
| `eval/queries-test.jsonl` | 10（回顾性 split） | 稳健性检查 |
| `data/public/queries.jsonl` | 25（RAGBench emanual） | 英文外部锚点 |

## 5. 评测协议

完整定义见 [`evaluation.md`](evaluation.md)。核心：

- `可答 acc` 与 `弃答 acc` 分母分离；所有 rung 同一可答分母。
- 主指标是 `evidence R@1` / `R@5` 和 `MRR@10`，不是 `doc R@5`。
- 比例指标报 Wilson 95% CI；bootstrap 保留在 `summary.json`。
- judge 做 30 条 A/B 字段对调校准（30/30 一致），但非人工标注。
- 综合分：`100 × (0.50×可答 acc + 0.30×evidence R@1 + 0.20×MRR@10) / (1 + 100×cold $/q)`。

## 6. 成本

| 项 | 金额 |
|---|---:|
| 语料生成（LLM 分类 1,000 篇） | $0.365 |
| 原 contextual + RAPTOR | ~$1.87 |
| masked contextual | ~$0.45 |
| 评测 + judge + 校准 + 公开锚点 | ~$1.1 |
| **合计** | **~$3.8 / $10** |

硅基流动的 bge-m3 / bge-reranker 免费，实际 ¥0。

## 7. 关键决策

| 决策 | 理由 | 代价 |
|---|---|---|
| 一张图 + 计划表 | 避免 9 套重复节点；评测和线上走同一份代码 | 拓扑不如静态图直观 |
| 固定 top-5 / 4,000 token | 公平比较 | 长文档会被截断 |
| 不做 GraphRAG | 1,000 篇上社区太薄；RAPTOR 用 1/10 成本覆盖全局题 | 全局题上限略低 |
| 不做 Web UI | 评测和报告优先 | 只能用 CLI / Markdown |
| SQLite 缓存 | 重复跑不重复付费 | 缓存键要管好 prompt 版本 |
| masked 前缀复用 | 省 45 分钟 / $1.4 | 前缀语义仍可能带原标题痕迹 |

## 8. 风险

| 风险 | 现状 / 对策 |
|---|---|
| 样本小 | 30 / 25 / 10 / 25；报 CI，不把 1–2 题差异当结论 |
| 标题泄漏 | masked 变体 + 干扰文档；仍非完全去偏 |
| judge 偏差 | A/B 对调；没有第二模型 / 人工标注 |
| RAPTOR 摘要不含 evidence | 改按「覆盖 gold 文档」算命中 |
| 公开锚点小 | 25 题英文手册；不能外推中文 |
| 成本失控 | 缓存 + 先 30 题 + cold $/q 看板 |

## 9. 目录

```text
src/babyrag/        # 核心库
scripts/            # 语料、索引、评测、重建
eval/               # 问题集
results/            # leaderboard / summary / per_query / cost
docs/               # 设计、结果、指标、里程碑
research/           # 实现前研究（有过期假设）
```
