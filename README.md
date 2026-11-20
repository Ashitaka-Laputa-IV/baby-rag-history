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
- **doc recall@5 全是 100%**：1,000 篇语料上「找对文档」已经饱和，差异全在 chunk 级排序与生成。

## 诚实的限制

- 首版评测集是 **30 题**（single 12 / entity 6 / multi 6 / global 3 / none 3）。global 与 none 各 3 题，
  只能看方向，不能下统计结论。
- **天花板效应仍在**：即使扩到 1,000 篇，`doc recall@5` 依旧 100%；真正拉开差距的是 chunk 级排序。
  要测出检索层的差异，需要更大的语料或去标题线索的难题集。
- 本项目演示的是**机制、成本与失败模式**，不是论文复现：DPR 训练、Self-RAG 的 reflection token、
  CRAG 的网页回退都换成了同机制的 LLM/规则替代。
- `N/A` 表示该档不具备该能力，不是 0 分。

## License

代码 MIT。语料 CC-BY-SA-4.0，衍生自中文维基百科，需署名并以相同方式共享。
