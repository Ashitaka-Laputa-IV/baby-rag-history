# baby-rag-history · 方案设计（待审批）

> 目标：用 **一个 LangGraph 状态机**，在同一份中文维基语料（**1,000 篇**）、同一批问题上，逐级演示 RAG 发展史上 9 个里程碑；每个里程碑只改一个变量，并给出可复算的成本。
>
> **已批准的执行选项：B —— 扩到 1,000 篇，用 rag-ladder 的 LLM 分类 + 嵌套抽样方法。**
>
> **本文件是审批稿。批准后才开始写实现代码。**

---

## 1. 一句话方案

把 RAG 发展史做成一把「梯子」：`r0 朴素 RAG → r1 混合检索 → r2 精排 → r3 HyDE → r4 多查询 → r5 上下文索引 → r6 层次摘要 → r7 自评纠错 → r8 Agentic 多轮`，外加一个 `adaptive` 路由模式。九个 rung 共用同一张 LangGraph、同一份 chunks、同一套 30 题核心评测（可扩展）；GraphRAG **不实现**，只作为「成本否决」案例写进报告。

---

## 2. 为什么砍掉 GraphRAG（成本论证）

| 项 | 数值 |
|---|---|
| 1,000 篇 / 4,812 chunks 的实体抽取 | 约 4,812 次 LLM 调用 |
| 估算（按 869 chunk 的 $1.114 线性外推） | **≈ $6.2**（索引约 $3.8 + 查询 $1.5 + 重试） |
| 占 $10 预算 | **约 62%**，仍买得起，但只剩很少余量 |
| 工程风险 | 实体格式解析失败、空图、Leiden 依赖、调试周期长；200 篇上社区结构太薄 |
| 结论 | **不做**。在报告中用一页说明「为什么全局问题不一定需要图」，并把预算让给 r6 RAPTOR（成本约 $0.08，覆盖同类全局题） |

> **重要更正**：GraphRAG 的排除是 **ROI / 排期决策，不是预算决策**。$1.114 在 $10 内完全可行；不做是因为它比 RAPTOR 贵约 14 倍、工程风险高，而在这 200 篇语料上社区结构太薄，收益无法兑现。报告必须这样写，避免「因为太贵所以不做」的错误归因。

---

## 3. 九个里程碑（Rung 设计）

| # | 名称 | 锚点 | 相对上一级只改了什么 | 主要解决 |
|---|---|---|---|---|
| R1 | Naive RAG | Lewis et al. 2020, arXiv:2005.11401 | 基线：dense top-5 → 生成 | 单跳事实 |
| R2 | Hybrid Retrieval | Robertson & Zaragoza 2009 (BM25) + Cormack et al. 2009 (RRF) | 加 BM25 一路，RRF(k=60) 融合 | 专名、编号、稀有词 |
| R3 | Rerank | Nogueira & Cho 2019, arXiv:1901.04085；bge-reranker-v2-m3 | 召回 50 → 精排 top-5 | 召回够但排序差 |
| R4 | HyDE | Gao et al. 2022 HyDE, arXiv:2212.10496 | 查询表示：embed「假答案」而不是原问题 | 问题与文档措辞不一致 |
| R5 | Multi-Query / RAG-Fusion | Rackauckas 2024, arXiv:2402.03367 | 查询数量：1 问变 3 问，RRF 融合 | 单词命中不稳 |
| R6 | Contextual Retrieval | Anthropic 2024-09 | 索引侧：每个 chunk 前加 50–100 token 文档内定位说明 | chunk 被切碎后丢语境 |
| R7 | RAPTOR | Sarthi et al. 2024, arXiv:2401.18059 | 加聚类摘要树，跨层检索 | 全局综述、跨文档总结 |
| R8 | Corrective / Self-Reflective | Yan et al. 2024 CRAG, arXiv:2401.15884；Asai et al. 2023 Self-RAG, arXiv:2310.11511 | 生成前加检索质量评估、改写重查、弃答 | 检索错了、该拒答 |
| R9 | Agentic ReAct | Yao et al. 2022, arXiv:2210.03629；综述 arXiv:2501.09136 | 把单次检索变成最多 4 步的 `Thought→Action→Observation` | 多跳、需要追问 |
| — | Adaptive Router | Jeong et al. 2024, arXiv:2403.14403 | 每题先分类，选上面某一档 | 不是新能力，是「何时用哪档」 |

代码内的稳定 id：`r0_dense → r1_hybrid → r2_rerank → r3_hyde → r4_multiquery → r5_contextual → r6_raptor → r7_crag → r8_agentic`，外加 `adaptive`。

**为什么是 9 个而不是 6 个**：6 个能覆盖主线，但 HyDE（改查询表示）和 Multi-Query（改查询数量）是两个正交旋钮，合并会看不清「改哪里」；Contextual（索引侧）与 RAPTOR（索引结构）也值得分开；CRAG（先判断再答）与 Agentic（边查边答）同样如此。9 个 rung 仍能把总成本压在 $3 左右。

**明确不做的**：GraphRAG（成本）、ColBERT/多向量（工程复杂度与收益不匹配）、微调/训练（预算与时间）。

---

## 4. 一份语料、一套题、一套指标（公平性纪律）

| 项 | 决定 |
|---|---|
| 语料 | **1,000 篇**，用 `rag-ladder/scripts/prepare_corpus.py --size 1000 --classifier llm` 生成（LLM 五主题分类 + 嵌套抽样；原 200 篇是它的前缀）。字段 `doc_id,title,text,theme,url,chars,license` |
| 切分 | `rag-ladder/scripts/build_chunks.py`（512 token / overlap 64），实测 **4,740 块**；超长句按 `fix_long_sentences.py` 预处理。**冻结不再切** |
| 问题 | 直接复制 `rag-ladder/eval/queries.jsonl`。**实际只有 30 题**：single 12 / entity 6 / multi 6 / global 3 / none 3；字段 `qid,type,theme,question,gold_answer,gold_doc_ids,evidence,should_abstain` |
| 上下文预算 | 所有 rung 统一 `MAX_CONTEXT_TOKENS = 4000`，超出截断 |
| 生成模型 | `deepseek-v4.1-flash`，`temperature=0`，同一 system prompt 骨架 |
| Embedding / Rerank | `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3`（硅基流动，免费） |
| 弃答阈值 τ | 在 dev 子集上标定一次，之后所有 rung 共用 |
| 指标 | `evidence_recall@1/3/5/10`（gold doc 是否进入上下文）、`hit@1`（排序质量）、`answer_correctness`（LLM judge，A/B 对调）、`abstain_accuracy` / `false_abstain_rate`、`citation_precision`、tokens / 调用数 / P95 延迟 / USD / `cost_per_correct` |

> **N/A 原则**：某 rung 做不到的题型标 `N/A`，不标 0。例如 r0 不做全局综述，就是 `N/A`。
>
> **样本量警告**：global 与 none 各只有 3 题，只适合演示方向，不能下统计结论。首版用这 30 题跑通；若需要更稳的数字，用 `scripts/expand_queries.py` 再合成约 60 题（成本约 $0.1），**核心 30 题与扩展集合分表报告**，绝不混在一起算平均。

### 4.1 天花板效应（必须正面处理）

实测基线：现有 30 题上，M1 已经 **29/30 正确**，`evidence_recall@5 = 27/27`；唯一失败是全局聚合题 q025。原因是：

- 全库只有 869 个 chunk，top-5 几乎必然覆盖 gold；
- 80.5% 的文档正文包含标题，标题词直接抬高了检索分数；
- 单跳 / 专名 / 多跳题都太容易，global 与 none 各只有 3 题。

**不做处理的后果**：9 个 rung 在表里全是 96%，读者会得出「RAG 发展史没有意义」的错误结论。

**对策（本期执行）**：

| 对策 | 做法 | 成本 |
|---|---|---|
| 去标题线索 | 评测时额外跑一遍「chunk 文本去掉标题词」的索引；对比 title-masked recall | $0（索引免费，仅重建一次） |
| 检索分层指标 | 同时报 Recall@1 / @3 / @5 / @10，不只报 @5 | $0 |
| 难题集 | 新增 `eval/queries-hard.jsonl`：20 题（6 道 3 跳、6 道 5+ 文档全局、4 道真未知实体、4 道同实体不同文档混淆），人工写或合成后人工筛 | ~$0.15 |
| 上下文压力测试 | 固定 gold chunk，前后插入 N 个无关 chunk，测 Lost-in-the-Middle | ~$0.05 |
| 诚实结论 | 主榜分「饱和集（30 题）」与「难题集（20 题）」两张表；饱和集用于演示机制，难题集用于比较能力 | $0 |

> 如果连难题集也无法拉开差距，就在 README 明确写：**在 200 篇语料上，RAG 的瓶颈已不是检索，而是语料规模与问题难度；本项目的价值是机制、成本与失败模式，不是排行榜。** 这比伪造提升更可信。

---

## 5. LangGraph 编排设计（核心）

### 5.1 一张图 + 计划驱动（plan → dispatch）

**关键决策：不是 9 张图，而是 1 张图 + 9 行计划表。** 每个里程碑是「有序阶段列表」的一行；`ladder` 模式查表，`adaptive` 模式运行时生成同一张表。两条路径走**完全相同**的节点代码，避免「评测一套、上线另一套」。

```text
START → intake → plan → dispatch ─┬→ retrieve_dense      ─┐
                                  ├→ retrieve_bm25       ─┤
                                  ├→ fuse_rrf            ─┤
                                  ├→ rerank              ─┤
                                  ├→ hyde                ─┤
                                  ├→ rewrite_multi_query ─┤
                                  ├→ expand_context      ─┤
                                  ├→ retrieve_raptor     ─┤
                                  ├→ grade               ─┼→ dispatch
                                  ├→ agentic (子图)       ─┤
                                  ├→ generate            ─┤
                                  ├→ verify              ─┤
                                  └→ finalize → END      ─┘
```

- `plan` 只产出 `plan: list[str]` 与 `cursor: int`，不执行阶段。
- `dispatch` 是唯一路由点：读 `plan[cursor]`，返回下一个节点名；阶段节点做完只返回 `cursor + 1`，再回 `dispatch`。
- **只有 Agentic ReAct 是子图**（它有自己的消息/工具循环）；其余阶段全部扁平化。
- 不在同一节点上同时用静态边和 `Command(goto=...)`（会两条路都跑）。

```python
RUNGS: dict[str, list[str]] = {
    "r0_dense":       ["retrieve_dense", "generate"],
    "r1_hybrid":      ["retrieve_hybrid", "fuse_rrf", "generate"],
    "r2_rerank":      ["retrieve_hybrid", "fuse_rrf", "rerank", "generate"],
    "r3_hyde":        ["hyde", "retrieve_hybrid", "fuse_rrf", "rerank", "generate"],
    "r4_multiquery":  ["rewrite_multi_query", "retrieve_multi", "fuse_rrf", "rerank", "generate"],
    "r5_contextual":  ["retrieve_contextual", "fuse_rrf", "rerank", "generate"],
    "r6_raptor":      ["retrieve_raptor", "rerank", "generate"],
    "r7_crag":        ["retrieve_hybrid", "fuse_rrf", "rerank", "grade", "generate"],
    "r8_agentic":     ["agentic", "verify", "generate"],
}
```

- `--rung all` 依次用 9 行计划跑同一批题。
- `adaptive` 先用便宜检索拿特征（top1 分数、BM25 命中数、问题长度/题型），规则优先；规则不确定才调一次分类 LLM（`temperature=0`，缓存），并记录 `routing_distribution`、`oracle_best_rung` 与 `regret`。
- **CRAG 的「反思→重查」不写进 plan**（会无限增长）：`grade` 节点内部完成「评分 → 改写 → 再检索一次」，只把最终结果写回 state；这样更好控成本。

### 5.2 State 设计

```python
class RAGState(TypedDict, total=False):
    # 输入 / 计划
    qid: str
    question: str
    mode: Literal["ladder", "adaptive"]
    rung: str
    plan: list[str]                # 本次要跑的有序阶段
    cursor: int
    route_reason: str
    # 查询侧
    query_variants: list[str]      # HyDE / multi-query 产物
    # 检索
    hits: list[Chunk]              # chunk_id, doc_id, text, score, source
    # 生成
    draft: str
    answer: str
    citations: list[str]
    # 自评 / 控制
    grade: Literal["correct", "ambiguous", "incorrect"]
    attempts: int
    max_attempts: int
    abstain: bool
    # 可观测
    node_records: Annotated[list[NodeRecord], operator.add]
    trajectory: list[Step]         # 每步 node / action / observation
    cost: CostAccumulator          # tokens_in/out, calls, latency, usd, cache_hits
    error: str | None
```

- **模型/索引不进 state**：`ChatOpenAI`、Chroma、BM25、reranker 放 `Runtime.context` 或模块级单例，避免 checkpointer 序列化失败。
- **持久化**：`MemorySaver` 用于单次评测；`SqliteSaver` 用于 REPL / 可回放；`recursion_limit` 放在顶层 config。
- **成本**：统一 `LLMClient` 包装器读取 `AIMessage.usage_metadata`，缺失时读 `response_metadata["token_usage"]`，再兜底 tiktoken 估算并标 `estimated`；opencode-go 的 `usage.opencode.list_cost` 优先。
- **流式**：CLI 用 `astream(stream_mode="updates", version="v2")`，把 `trajectory` 逐步打印。

### 5.3 节点职责

| 节点 | 做什么 | 是否花 LLM |
|---|---|---|
| `intake` | 校验问题、生成 session id、初始化 cost | 否 |
| `plan` | ladder 查 `RUNGS` 表；adaptive 生成计划 | 否 / 1 次小调用 |
| `dispatch` | 纯函数：读 `plan[cursor]` 返回下一个节点 | 否 |
| `hyde` | 生成「假答案」并 embed | 1 次 |
| `rewrite_multi_query` | 生成 3 条查询变体 | 1 次 |
| `retrieve_dense` | bge-m3 + Chroma top-k | 否（embedding 免费） |
| `retrieve_bm25` | jieba + bm25s top-k | 否 |
| `retrieve_hybrid` | 并行调用 dense + bm25 | 否 |
| `retrieve_multi` | 对 3 条变体分别检索 | 否 |
| `fuse_rrf` | RRF(k=60) 融合并截断 | 否 |
| `rerank` | bge-reranker-v2-m3 精排 top-5 | 否（rerank 免费） |
| `retrieve_contextual` | r5：走 contextual 向量 + contextual BM25 | 否 |
| `expand_context` | 备用：把上下文前缀拼回普通 chunk | 否（索引期已生成） |
| `retrieve_raptor` | r6：跨层检索摘要树（collapsed tree） | 否 |
| `grade` | CRAG 三分类；**批量评 5–8 个 chunk**，低置信时内部改写重查一次 | 1–2 次小调用 |
| `generate` | 基于 4000 token 上下文生成 + 引用 | 1 次 |
| `verify` | 引用合法性、是否该弃答 | 否（规则） |
| `agentic` | ReAct 子图，工具 `search(query)`，`max_steps=4` | 每步 1 次 |
| `finalize` | 汇总 cost、写 per-query 结果 | 否 |

---

## 6. 成本模型（硬预算 $10）

### 6.1 单价

| 项 | 价格 |
|---|---|
| DeepSeek V4.1 Flash 输入 | $0.15 / 1M tokens |
| DeepSeek V4.1 Flash 输出 | $0.60 / 1M tokens |
| 缓存读 | $0.003 / 1M tokens |
| bge-m3 / bge-reranker-v2-m3 | ¥0（硅基流动免费） |

> **计价口径**：全部用 `tiktoken` 实测 token，不用「1.5 字/token」。实测中文 cl100k 为 **0.877 字/token（1.14 token/字）**；固定 prompt 逐条 tokenize。详见 `research/cost-model.md` 与可复算脚本 `research/cost_model.py`。

### 6.2 索引成本（一次性，实测口径）

| 项 | 调用数 | 输入 tokens | 输出 tokens | 成本 |
|---|---:|---:|---:|---:|
| r5 Contextual 前缀（4,740 chunk） | 4,740 | 约 10.6M | 约 355k | **≈ $0.89** |
| r6 RAPTOR 摘要树（527 L1 + 14 L2） | 541 | 约 2.0M | 约 135k | **≈ $0.39** |
| 其余 rung | 0 | 0 | 0 | $0 |
| **合计** | **5,407** | **约 13M** | **约 510k** | **≈ $1.33** |

### 6.3 评测成本（首版 30 题 × 一次，按实测每 150 题 ÷5）

| Rung | 每题成本（150 题口径） | 每题成本 | 30 题 |
|---|---:|---:|---:|
| r0 Naive | $0.000430 | $0.000430 | $0.0129 |
| r1 Hybrid（rerank 免费，无额外 LLM） | $0.000430 | $0.000430 | $0.0129 |
| r2 Rerank | $0.000430 | $0.000430 | $0.0129 |
| r3 HyDE | $0.000626 | $0.000626 | $0.0188 |
| r4 Multi-Query | $0.000494 | $0.000494 | $0.0148 |
| r5 Contextual | $0.000487 | $0.000487 | $0.0146 |
| r6 RAPTOR | $0.000468 | $0.000468 | $0.0140 |
| r7 CRAG（评分 + 30% 重试） | $0.001031 | $0.001031 | $0.0309 |
| r8 Agentic（均 3.2 步） | $0.001238 | $0.001238 | $0.0371 |
| r9 Adaptive（离线路由，复用缓存答案） | — | $0.000018 | $0.0005 |
| **合计** | | | **≈ $0.169** |

> **关键发现**：r1 Hybrid 与 r0 的 LLM 成本**完全相同**（BM25 与 rerank 免费，且不增加生成调用）；r7 是 r0 的 **2.4 倍**（不是旧文档写的 1.4 倍）。扩展题量是线性的：90 题 ≈ $0.51，150 题 ≈ $0.85。

### 6.4 总预算

| 阶段 | 内容 | 预算 |
|---|---|---:|
| 索引 | r5 + r6（1,000 篇口径） | $1.33 |
| 开发迭代 | 5 轮 × 30 题 × 平均 $0.0005 | $0.075 |
| 正式评测 | 30 题 × 10 档 | $0.18 |
| LLM judge | 30 × 9 档 × A/B 两遍 = 540 次 | $0.042 |
| 重试 / 缓存未命中缓冲 | 30% | $0.49 |
| **合计（首版 30 题）** | | **≈ $2.12** |
| 可选：题目扩展到 150 题 | 评测与 judge ×5 | +$1.1 |
| **可用预算** | | **$10.00** |
| **剩余** | | **≈ $7.88** |

> 成本模型的最坏情景（S7 失败率 50%、S8 5 步、重试翻倍）为 **$4.56**，仍只占预算 46%。**预算不是本项目的约束，评测噪声与天花板效应才是。**

硅基流动：embedding 869 块 + 30 题 + 可能的消融，全部用免费档，**¥0**；¥10 额度原封不动。

> 即使把 200 篇扩到 2,000 篇（索引与评测约 ×10），仍能压在 $10 内；但本期不做，保持 200 篇以便快速迭代。

---

## 7. 目录结构（uv 管理）

```text
baby-rag-history/
├─ opencode.json                 # LangChain MCP（已配置）
├─ pyproject.toml                # uv 项目
├─ uv.lock
├─ .env.example
├─ README.md                     # 结论 + 排行榜 + 复现步骤
├─ docs/design.md                # 本文件
├─ research/                     # 子智能体调研结果
├─ data/                         # git-ignored 大文件
│   ├─ corpus.jsonl              # 从 rag-ladder 复制
│   ├─ chunks.jsonl
│   ├─ chroma/                   # 向量索引
│   └─ cache/                    # SQLite 缓存
├─ eval/queries.jsonl            # 从 rag-ladder 复制
├─ src/babyrag/
│   ├─ settings.py               # 环境变量 + 单价
│   ├─ llm.py                    # ChatOpenAI + x-opencode-session
│   ├─ embeddings.py             # bge-m3
│   ├─ store.py                  # Chroma
│   ├─ bm25.py                   # jieba + bm25s
│   ├─ types.py                  # State / Hit / Cost
│   ├─ cache.py                  # SQLite 缓存
│   ├─ rungs.py                  # RUNGS 计划表（r0–r8）
│   ├─ graph.py                  # build_graph
│   ├─ nodes/                    # 每个节点一个文件
│   └─ eval/                     # metrics, judge, leaderboard
├─ scripts/
│   ├─ prepare_data.py           # 复制并校验数据
│   ├─ build_index.py            # 向量 + BM25
│   ├─ build_contextual.py       # r5 索引
│   ├─ build_raptor.py           # r6 索引
│   └─ run_eval.py               # --rung all
└─ tests/
```

**依赖**：`langgraph`, `langchain-core`, `langchain-openai`, `langchain-chroma`, `chromadb`, `bm25s`, `jieba`, `tiktoken`, `pydantic`, `python-dotenv`, `typer`, `rich`, `pytest`。

---

## 8. 关键实现约定

### 8.1 OpenCode Go 接入（来自 baby-langgraph-code-agent 的文档）

- 环境变量：`OPENCODE_GO_API_KEY`（回退 `OPENCODE_API_KEY`）、`OPENCODEGO_BASE_URL`、`OPENCODEGO_MODEL`。
- **每个请求必须带 `x-opencode-session`**，否则 HTTP 400。
- 会话 id 每进程生成一次 `ses_<uuid hex>`，跨轮次复用，以命中提示词缓存。
- `ChatOpenAI(base_url=..., default_headers={"x-opencode-session": sid}, temperature=0)`。

### 8.2 缓存（把成本压到 $3 的关键）

SQLite 表 `llm_cache(key TEXT PRIMARY KEY, value TEXT, model TEXT, created_at TEXT)`：

```text
key = sha256(stage | prompt_version | model | normalized_input)
```

缓存对象：r5 上下文前缀、HyDE 假设文档、multi-query 改写、r6 摘要、r7 评级、judge 评分。重复跑同一题不再花钱。

### 8.3 数据准备（实际执行）

1. `prepare_corpus.py --size 1000 --classifier llm --workers 24`：LLM 五主题分类 + 嵌套抽样，得到 1,000 篇均衡语料（成本 $0.365，70 分钟）。
2. **种子合并**：LLM 分类有非确定性，原 200 篇只保留了 185 篇、丢了 3 个 gold 文档；用 `merge_corpus.py` 把原 200 篇作为种子、从新语料补 800 篇到 1,000，确保 30 题全部可评测。
3. `fix_long_sentences.py`：拆掉 5 个超长句，避免 chonkie 产出 >512 token 的块。
4. `build_chunks.py`：512/64，得到 4,740 块。
5. `queries.jsonl` 只有 `gold_doc_ids`；评测时用 `doc_id` + `evidence` 子串定位 gold chunk。
6. `data/` 进 `.gitignore`（CC-BY-SA + 索引体积）；README 保留来源与署名。

### 8.4 评测执行顺序

1. `prepare_data` → 校验 200/869/30 数量。
2. `build_index`（dense + BM25）→ 免费。
3. `build_contextual` / `build_raptor` → 约 $0.24，先跑 20 chunk 试点再全量。
4. smoke：3 题 × R1，确认引用与成本字段。
5. dev：30 题 × 9 档，调 τ 与 prompt。
6. full：30 题 × 10 档 + judge（可选：先 `expand_queries --size 60` 扩到 90 题）。
7. 生成 `results/leaderboard.md` 与 `results/cost.jsonl`。

---

## 9. 交付物

| 交付物 | 说明 |
|---|---|
| `README.md` | 一页看懂：10 档是什么、排行榜、成本、如何复现 |
| `results/leaderboard.md` | 两张表：饱和集（30 题）与难题集（20 题）；题型 × rung 的准确率 / Recall@k / 弃答 / 成本 / P95 |
| `results/retrieval.md` | Recall@1/3/5/10 + title-masked 对比 + Lost-in-the-Middle 压力测试 |
| `results/cost.jsonl` | 每次调用的 tokens 与 USD，可逐条审计 |
| `eval/queries-hard.jsonl` | 20 道难题（3 跳 / 全局 / 未知实体 / 混淆） |
| `docs/history.md` | 每个里程碑的 1 页说明：论文、机制、在本项目的表现 |
| `docs/graphrag-not-built.md` | 成本否决的完整论证 |
| 可运行 CLI | `uv run babyrag ask --rung r5 "问题"`、`uv run babyrag eval --rung all` |

---

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| LLM 返回的 JSON / 引用格式不稳 | 每个节点带 Pydantic 校验 + 一次重试；失败降级为规则路径 |
| 中文 BM25 分词不一致 | 索引与查询共用同一个 jieba 实例与停用词表 |
| RAPTOR 聚类在 200 篇上过碎 | 先固定 8–10 块/簇、2 层；不调参 |
| r7 评估器过度保守 | 同时报「误弃率」，τ 只在 dev 集标定一次 |
| r8 死循环 | 硬上限 `max_steps=4` + 总 token 上限 6000 |
| 预算被重跑吃光 | SQLite 缓存 + 先 30 题 + 成本看板 |
| Windows 路径 / Chroma 锁 | 统一 `pathlib.Path`；索引构建单进程 |

---

## 11. 待你审批的决策点

1. **9 个 rung 的取舍**是否认可？特别是 HyDE 与多查询分成两档、Contextual 与 RAPTOR 分成两档。
2. **GraphRAG 只写报告不实现**，是否同意？
3. **天花板效应怎么处理**（这是最关键的一项）：
   - **A（推荐）**：保留 200 篇语料，新增 20 道难题 + 去标题索引 + Recall@1/3/5/10，主榜拆成「饱和集 / 难题集」；
   - **B**：从 `zhwiki-00002.parquet` 扩到 1,000–2,000 篇，让检索真正有难度（成本仍 < $5，但构建时间从分钟级变成小时级）；
   - **C**：不处理，接受「全 96%」的演示，只在 README 说明原因。
4. **首版用现有 30 题**，还是同时把题目扩到 90 题？
5. 是否需要 **Web UI**（Streamlit/Gradio）？默认不做，只做 CLI + Markdown 报告。
6. 是否把 `data/` 完全 gitignore（推荐），只在 README 写来源？

---

*批准后我将按此执行：uv 建环境 → 复制数据 → 索引 → 9 个 rung → 评测 → 排行榜 → 提交 PR。*

---

## 12. 实测结果（2026-11-20）

| rung | acc | chunk hit@5 | abstain | cold $/q |
|---|---:|---:|---:|---:|
| r0 Naive | 88.9% | 96.3% | N/A | $0.00030 |
| r1 Hybrid | 88.9% | 96.3% | N/A | $0.00030 |
| r2 Rerank | 96.3% | 96.3% | N/A | $0.00030 |
| r3 HyDE | 96.3% | 96.3% | N/A | $0.00081 |
| r4 Multi-query | 92.6% | 96.3% | N/A | $0.00046 |
| r5 Contextual | **100.0%** | **100.0%** | N/A | $0.00031 |
| r6 RAPTOR | 96.3% | 96.3% | N/A | $0.00030 |
| r7 CRAG | 93.3% | 92.6% | **100.0%** | $0.00065 |
| r8 Agentic | 90.0% | 96.3% | **100.0%** | $0.00086 |
| adaptive | 90.0% | 96.3% | **100.0%** | $0.00045 |

结论：**r5 contextual 与 r2 rerank 是本期真正的赢家**；HyDE/multi-query 在中文维基上没有增益；
GraphRAG 的排除不影响上述结论（RAPTOR 已覆盖全局题）。`doc recall@5` 在 1,000 篇上仍为 100%，
说明「找对文档」已饱和，差异在 chunk 级排序。
