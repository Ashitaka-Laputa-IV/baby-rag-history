# baby-rag-history

**RAG 发展史的九个里程碑，用一张 LangGraph 状态机跑在同一份中文维基语料上。**

不是九个互不相干的 notebook，而是一个 `plan → dispatch` 图：每个里程碑是一行「阶段计划」，
共用同一批 chunks、同一套问题、同一套指标。每个 rung 只改一个变量。

> 设计稿：`docs/design.md` · 调研：`research/` · 里程碑说明：`docs/history.md`

## 九个 rung

| rung | 里程碑 | 只改了什么 | 锚点 |
|---|---|---|---|
| `r0_dense` | Naive RAG | dense top-5 → 生成 | Lewis et al. 2020, arXiv:2005.11401 |
| `r1_hybrid` | Hybrid + RRF | 加 BM25 一路 | Cormack et al. 2009 |
| `r2_rerank` | Rerank | 召回 20 → bge-reranker 取 5 | Nogueira & Cho 2019, arXiv:1901.04085 |
| `r3_hyde` | HyDE | embed「假答案」而非问题 | Gao et al. 2022, arXiv:2212.10496 |
| `r4_multiquery` | Multi-Query | 1 问变 3 问 + RRF | Rackauckas 2024, arXiv:2402.03367 |
| `r5_contextual` | Contextual Retrieval | 索引期给 chunk 补定位说明 | Anthropic 2024-09 |
| `r6_raptor` | RAPTOR | 聚类摘要树，跨层检索 | Sarthi et al. 2024, arXiv:2401.18059 |
| `r7_crag` | CRAG / Self-RAG | 检索评估 + 改写重查 + 弃答 | Yan et al. 2024, arXiv:2401.15884 |
| `r8_agentic` | Agentic ReAct | 多轮 `SEARCH → OBSERVATION` | Yao et al. 2022, arXiv:2210.03629 |
| `adaptive` | Adaptive Router | 按题型/长度选上面一档 | Jeong et al. 2024, arXiv:2403.14403 |

**GraphRAG 不实现**。不是买不起（实测约 $1.11），而是 ROI/排期：在 1,000 篇语料上社区结构很薄，
r6 RAPTOR 用约 1/14 的成本覆盖了全局综述能力。完整论证见 `docs/graphrag-not-built.md`。

## 架构

```text
START → plan → dispatch ─┬→ retrieve_dense / retrieve_hybrid / retrieve_multi
                         ├→ hyde / rewrite_multi_query / fuse_rrf / rerank
                         ├→ expand_context / retrieve_raptor / grade
                         ├→ agentic (ReAct 子图) / verify / generate
                         └→ finalize → END
```

- `ladder` 模式：`plan = RUNGS[rung]`，固定台阶。
- `adaptive` 模式：规则优先，按题型选档。
- 只有 Agentic 是子图；其余阶段扁平化。
- 所有 LLM 调用走 SQLite 缓存（键 = stage + prompt_version + model + 输入哈希），重跑不重复付费。

## 数据

- 来源：`wikimedia/wikipedia` 20231101 中文 shard（`zhwiki-00002.parquet`），经
  `scripts/prepare_corpus.py` 清洗、LLM 五主题分类、嵌套抽样。
- 规模：**1,000 篇**（原 200 篇是它的前缀），512/64 切分。
- 许可：**CC-BY-SA-4.0**，署名见每篇 `corpus.jsonl` 的 `url` 字段。
- `data/` 已 gitignore；仓库只保留脚本与 manifest。首次使用按下方步骤重建。

## 快速开始

```bash
uv sync
cp .env.example .env          # 填 OPENCODE_GO_API_KEY 与 SILICONFLOW_API_KEY

# 1) 语料：prepare_corpus 需要 data/raw/zhwiki-00002.parquet（约 127 MB）
uv run python -m scripts.prepare_corpus --size 1000 --classifier llm --workers 24
uv run python -m scripts.fix_long_sentences
uv run python -m scripts.build_chunks

# 2) 索引
uv run python -m scripts.build_index
uv run python -m scripts.build_contextual --index   # r5，约 1,000 次 LLM 调用
uv run python -m scripts.build_raptor               # r6

# 3) 单题
uv run babyrag ask "增禄的父亲咸成曾任什么官职？" --rung r2_rerank

# 4) 全台阶评测
uv run python -m scripts.run_eval --rungs all --workers 4
```

## 成本

| 项 | 实测/估算 |
|---|---:|
| 语料生成（LLM 分类 1,000 篇） | $0.365 |
| 索引（r5 + r6，4,740 chunks） | 约 $1.33 |
| 30 题 × 10 档（冷启动） | 约 $0.14 |
| LLM judge | 约 $0.04 |
| **合计** | **约 $1.9 / $10** |
| 预算 | **$10**（OpenCode Go）· **¥10**（SiliconFlow，实际 ¥0） |

成本逐条写在 `results/cost.jsonl`；OpenCode Go 响应里的 token 用量优先，缺失时才用 tiktoken 估算。

## 结果（30 题，1,000 篇 / 4,740 chunks）

| rung | 里程碑 | acc | chunk hit@5 | abstain | cold $/q |
|---|---|---:|---:|---:|---:|
| `r0_dense` | Naive RAG | 88.9% | 96.3% | N/A | $0.00030 |
| `r1_hybrid` | Hybrid + RRF | 88.9% | 96.3% | N/A | $0.00030 |
| `r2_rerank` | + bge-reranker | 96.3% | 96.3% | N/A | $0.00030 |
| `r3_hyde` | HyDE | 96.3% | 96.3% | N/A | $0.00081 |
| `r4_multiquery` | Multi-query | 92.6% | 96.3% | N/A | $0.00046 |
| `r5_contextual` | Contextual Retrieval | **100.0%** | **100.0%** | N/A | $0.00031 |
| `r6_raptor` | RAPTOR | 96.3% | 96.3% | N/A | $0.00030 |
| `r7_crag` | CRAG + abstain | 93.3% | 92.6% | **100.0%** | $0.00065 |
| `r8_agentic` | Agentic ReAct | 90.0% | 96.3% | **100.0%** | $0.00086 |
| `adaptive` | Adaptive router | 90.0% | 96.3% | **100.0%** | $0.00045 |

完整表（含 p50/p95、doc recall@1/3/5/10）在 `results/leaderboard.md`；逐条成本在 `results/cost.jsonl`。

**读法**：

- **r5 contextual 是最大赢家**：索引期补语境后，chunk 级排序从 96.3% 到 100%，查询期不多花钱。
- **r2 rerank 是性价比最高的一档**：成本与 r0 相同（rerank 免费），准确率 +7.4pp。
- **HyDE / multi-query 在本语料上没有增益**：假答案会编造年份，反而把正确答案挤掉——这是真实代价。
- **r7/r8 的 abstain 都是 100%**：3 道无答案题全部正确拒答；但整体 acc 被多跳题拖低。
- **doc recall@5 全是 100%**：原始语料上「找对文档」已经饱和，差异全在 chunk 级排序与生成。

### 去标题后的差距（masked）

把 chunk 里的标题替换成「（标题已隐藏）」、另加 200 篇同姓/同类型干扰文档后，差距明显拉开：

| rung | acc | doc@1 | chunk@1 | chunk@3 | chunk@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| `r0_dense` | 70.4% | 81.5% | 44.4% | 59.3% | 63.0% | 0.526 | 0.552 |
| `r1_hybrid` | 77.8% | 81.5% | 44.4% | 70.4% | 70.4% | 0.556 | 0.594 |
| `r2_rerank` | 74.1% | 81.5% | 55.6% | 66.7% | 66.7% | 0.605 | 0.621 |
| `r3_hyde` | 77.8% | 81.5% | 29.6% | 55.6% | 70.4% | 0.443 | 0.508 |
| `r4_multiquery` | 70.4% | 85.2% | 55.6% | 66.7% | 66.7% | 0.605 | 0.621 |
| `r5_contextual` | 77.8% | 85.2% | **59.3%** | **70.4%** | **70.4%** | **0.642** | **0.658** |
| `r6_raptor` | 66.7% | 81.5% | 55.6% | 63.0% | 63.0% | 0.586 | 0.597 |
| `r7_crag` | 73.3% | 81.5% | 55.6% | 63.0% | 63.0% | 0.586 | 0.597 |
| `r8_agentic` | **80.0%** | 81.5% | 44.4% | 70.4% | 70.4% | 0.556 | 0.594 |
| `adaptive` | 76.7% | 81.5% | 44.4% | 70.4% | 70.4% | 0.556 | 0.594 |

对比原始语料：

| 指标 | 原始 | masked |
|---|---:|---:|
| doc@1 | 100% | 81.5–85.2% |
| chunk@1 | 74.1–85.2% | 29.6–59.3% |
| acc | 88.9–100% | 66.7–80.0% |

**结论**：真正的区分度来自 **chunk@1 / MRR**，而不是 doc R@5。r5 contextual 在排序指标上最强；r8 agentic 靠多轮检索拿到最高答案准确率，但首轮命中并不高。

完整表：`results/masked/leaderboard.md`。

### 手写难题集（hard）

原本写了 29 道，但复核发现 **4 道指代不清**（`h010/h011/h021/h022` 用「该市镇/该师团」导致所有 rung 都无法定位），已剔除。下表是 **25 道公平题**，每题 evidence 都校验过落在 gold 文档内。原索引上：

| rung | acc | chunk@1 | chunk@3 | MRR@10 |
|---|---|---:|---:|---:|
| `r0_dense` | 92.0% | 68.0% | 92.0% | 0.797 |
| `r1_hybrid` | 92.0% | 64.0% | 92.0% | 0.777 |
| `r2_rerank` | 92.0% | 80.0% | 96.0% | 0.881 |
| `r3_hyde` | 88.0% | 72.0% | 88.0% | 0.803 |
| `r4_multiquery` | **96.0%** | **84.0%** | **100.0%** | **0.913** |
| `r5_contextual` | **96.0%** | **84.0%** | 96.0% | 0.901 |
| `r6_raptor` | 92.0% | 80.0% | 96.0% | 0.881 |
| `r7_crag` | 92.0% | 84.0% | 96.0% | 0.901 |
| `r8_agentic` | 92.0% | 64.0% | 92.0% | 0.777 |
| `adaptive` | 92.0% | 64.0% | 92.0% | 0.777 |

完整表：`results/hard/leaderboard.md`；去标题 + 难题：`results/masked-hard/leaderboard.md`。

### 三套对比（怎么读）

| 设置 | acc 范围 | chunk@1 范围 | 谁赢 |
|---|---|---|---|
| 原题 + 原索引 | 88.9–100% | 48.1–85.2% | r5（acc 100%） |
| 原题 + masked | 66.7–80.0% | 29.6–59.3% | r8 acc、r5 排序 |
| 难题（25）+ 原索引 | 88.0–96.0% | 64.0–84.0% | r4/r5 |
| 难题（25）+ masked | 84.0–96.0% | 64.0–80.0% | r4 |

### 综合分（效果 × 成本）

公式：`综合 = 100 × (0.50×acc + 0.30×chunk@1 + 0.20×MRR@10) / (1 + 100×cold $/q)`

| rung | 原题 | masked | 难题（25） | 难题 + masked |
|---|---:|---:|---:|---:|
| `r0_dense` | 79.2 | 57.2 | 79.6 | 77.2 |
| `r1_hybrid` | 80.6 | 61.4 | 78.1 | 77.2 |
| `r2_rerank` | 84.6 | 63.8 | 84.6 | 83.3 |
| `r3_hyde` | 69.5 | 52.3 | 74.6 | 73.4 |
| `r4_multiquery` | 82.7 | 61.0 | 86.8 | **84.2** |
| `r5_contextual` | **89.5** | **66.9** | **88.1** | 83.4 |
| `r6_raptor` | 86.0 | 59.8 | 84.6 | 83.3 |
| `r7_crag` | 81.5 | 59.9 | 82.7 | 79.7 |
| `r8_agentic` | 77.0 | 59.0 | 73.5 | 71.8 |
| `adaptive` | 81.7 | 59.9 | 78.1 | 77.2 |

**综合看**：`r5 contextual` 赢 3/4 套，`r4 multi-query` 赢「难题 + masked」；`r2 rerank` 稳定前三，是最低成本的高分档；`r3 HyDE`、`r8 agentic`、`adaptive` 综合分偏低。

**真正稳定的结论**：`r2 rerank` 是低成本高收益；`r5 contextual` 在排序指标上稳定靠前；`r3 HyDE` 在这份中文维基语料上经常**反噬**（假答案把正确 chunk 挤掉）；`r8 agentic` 只在原题上靠多轮检索拿到最高 acc，遇到去标题/难题时优势消失。

## 诚实的限制

- 首版评测集是 **30 题**（single 12 / entity 6 / multi 6 / global 3 / none 3）。global 与 none 各 3 题，
  只能看方向，不能下统计结论。
- 手写难题集最终是 **25 道公平单跳改写题**（另有 4 道因指代不清被剔除），没有覆盖多跳/全局；它只回答「去掉标题后差距是否拉开」，不回答「Agentic 是否更强」。
- masked 的 contextual 前缀是**从原前缀派生的近似**：把标题字符串替换后重新嵌入，没有重新跑 5,676 次 LLM 生成。
  这样省了约 45 分钟和 $1.4，但前缀语义仍可能带有原标题的痕迹。
- **天花板效应仍在**：即使扩到 1,000 篇，`doc recall@5` 依旧 100%；真正拉开差距的是 chunk 级排序。
  要测出检索层的差异，需要更大的语料或去标题线索的难题集。
- 本项目演示的是**机制、成本与失败模式**，不是论文复现：DPR 训练、Self-RAG 的 reflection token、
  CRAG 的网页回退都换成了同机制的 LLM/规则替代。
- `N/A` 表示该档不具备该能力，不是 0 分。

## License

代码 MIT。语料 CC-BY-SA-4.0，衍生自中文维基百科，需署名并以相同方式共享。
