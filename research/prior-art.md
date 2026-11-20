# RAG 技术史教学项目 · 先行研究

> 核实口径：✅ = 2026-11-20 实际抓取 GitHub API / 页面确认；🟡 = 我的判断或估计，非仓库事实。星数为当日快照。

## 0. 一句话结论

现有多数"RAG 技术合集"是**平铺的笔记本目录**（一技术一 notebook），缺少"能力增量"叙事；LangGraph 官方的四种 agentic RAG 教程则各自独立、互不引用。本项目的机会是：**用同一份语料、同一个 LangGraph 状态机、同一套评测，把 8 个里程碑串成一条可比较的阶梯**，并把成本与"做不到什么"一起写进结果。

---

## 1. NirDiamant/RAG_Techniques ✅

| 事实 | 值 |
|---|---|
| Stars / Forks | 30,443 / 3.7k |
| 规模 | README 自称 **42+ notebooks**（表格编号到 38，另有新增条目） |
| 组织方式 | 按 **8 个类别**分组：Foundational、Query Enhancement、Context Enrichment、Advanced Retrieval、Iterative、Evaluation、Explainability、Advanced Architecture；每技术一个 notebook，配 `all_rag_techniques_runnable_scripts/`、`data/`、`evaluation/` |
| 形态 | 教学 notebook 合集，不是可组合的库；无统一 state / 无统一评测入口 |
| 最近更新 | 2026-11-02 |

各类别代表技术（✅ 均见于 README 表格）：Basic RAG、Chunk Size、Proposition Chunking；Query Transformations、HyDE、HyPE；Contextual Headers、Semantic Chunking、Contextual Compression；Fusion、Rerank、Hierarchical Indices；Adaptive Retrieval、Feedback Loop；DeepEval、GroUSE、End-to-End Eval；GraphRAG（LangChain / Microsoft / LightRAG）、RAPTOR、Agentic RAG、Self-RAG、CRAG、MemoRAG。

**"最常被引用"要分清两件事**：仓库本身没有提供论文引用计数，我无法核实哪一篇 notebook 学术引用最高（🟡）。可核实的是：其 README 明确挂论文链接、且在社区讨论中反复出现的是 **GraphRAG、RAPTOR、Self-RAG、CRAG、HyDE、Agentic RAG** 六项（✅ 链接存在；热度为 🟡）。

**最适合做阶梯的 6–8 项**（🟡 判断，标准：有原始论文 + 有独立 notebook + 相对上一级只改一个机制）：Basic RAG → RAPTOR → Multi-Query/RAG-Fusion → HyDE → Rerank → Adaptive Retrieval → CRAG → Self-RAG。这 8 项刚好覆盖"索引 / 检索 / 查询翻译 / 路由 / 迭代"五类，且每项都能用一个新增节点或一条新增条件边表达。

---

## 2. langchain-ai/rag-from-scratch（Lance Martin）✅

| 事实 | 值 |
|---|---|
| Stars / Forks | 10,133 / 2.3k |
| 形态 | **5 个 notebook，编号 1–18**（1–4、5–9、10–11、12–14、15–18 各一册） |
| 最近 push | 2025-06-26（已基本停更） |
| 配套 | YouTube playlist `PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x` |

| 集数 | 主题 | 归属 |
|---|---|---|
| 1–4 | Overview / Indexing / Retrieval / Generation | 基础三阶段 |
| 5–9 | Multi-Query、RAG-Fusion、Decomposition、Step-back、HyDE | Query Translation |
| 10 | Logical routing、Semantic routing | Routing |
| 11 | Query Construction（metadata filter / SQL / Cypher） | Query Structuring |
| 12–14 | Multi-Representation、RAPTOR、ColBERT | Indexing 深化 |
| 15–16 | CRAG、Self-RAG（✅ 镜像 notebook 文件名可核实） | Iterative / self-reflective |
| 17–18 | 合并于 `15_to_18.ipynb`，本次未逐节核实 | — |

**Canonical taxonomy（✅ 前六项由目录直接支撑；最后一项是教学归纳 🟡）**：

```
Indexing ─┐
Retrieval ├─ 基础三阶段（Part 1–4）
Generation┘
Query Translation   （Part 5–9：多查询/融合/分解/退步/HyDE）
Routing             （Part 10：逻辑路由 / 语义路由）
Query Structuring   （Part 11：把自然语言变成带过滤条件的结构化查询）
Iterative / Self-reflective（Part 15–16：CRAG / Self-RAG，后加）
```

这套六格分类是本项目最好的"坐标系"：**每个 rung 必须落在某一格，并且只能新增一格的能力**。注意它和 RAG Survey 的 Naive → Advanced → Modular 三范式是互补的：三范式讲宏观史，六格讲工程改动点。

---

## 3. LangGraph 官方 / 半官方 RAG 拓扑 ✅

四者本质是**同一个状态机的不同条件边**，差异只在"循环回到哪里"。

### 3.1 Agentic RAG（官方教程）
URL: https://docs.langchain.com/oss/python/langgraph/agentic-rag

```
START → generate_query_or_respond ─(有 tool_calls)→ retrieve(ToolNode)
                                  └(无)→ END
retrieve → grade_documents ─(relevant)→ generate_answer → END
                           └(irrelevant)→ rewrite_question → generate_query_or_respond
```
要点：`MessagesState`；是否检索由模型自己决定（`bind_tools`）；评分节点做二分类 relevance。这是"检索变成工具"的转折点。

### 3.2 Adaptive RAG
论文 https://arxiv.org/abs/2403.14403（2024, NAACL）；可运行教程 https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph/02-structures/07-langgraph-adaptive-rag

```
START → route_question ─(vectorstore)→ retrieve → grade_documents
                        │                          ├(generate)→ generate
                        │                          └(transform)→ transform_query → web_search
                        └(web_search)→ web_search ─────────────────→ generate
generate → grade_generation ─(grounded & useful)→ END
                             └(否)→ transform_query
```
要点：**入口就是条件边**（`set_conditional_entry_point`），按问题复杂度选择"不检索 / 单步 / 多步"。

### 3.3 Corrective RAG（CRAG）
论文 https://arxiv.org/abs/2401.15884（2024）；LangChain 博客 https://blog.langchain.com/agentic-rag-with-langgraph/（2024-02-07）；notebook `langchain-ai/langgraph/examples/rag/langgraph_crag.ipynb`

```
START → retrieve → grade_documents ─(relevant)→ generate → END
                                  └(irrelevant/ambiguous)→ transform_query → web_search → generate
```
官方 notebook 做了简化：**跳过 paper 的 knowledge-strips 精炼**；只要有文档不相关就补网搜。

### 3.4 Self-RAG
论文 https://arxiv.org/abs/2310.11511（2023-10）；同上博客与 `langgraph_self_rag.ipynb`

```
START → retrieve → grade_documents ─(任一 relevant)→ generate
                                  └(全 irrelevant)→ transform_query → retrieve
generate → grade_generation ─(有幻觉)→ generate
                             └(不 useful)→ transform_query → retrieve
                             └(pass)→ END
```
原论文用 `Retrieve / ISREL / ISSUP / ISUSE` 四种 reflection token 训练模型；LangGraph 版用 prompt 模拟，并把"每 chunk 各生成一次"简化为"全量上下文生成一次"。

> 教学含义（🟡）：这四张图可以只用一个 `GraphState` + 一个节点注册表实现；rung 的差别是**边表**，不是代码库。

---

## 4. 其他值得纳入"里程碑叙事"的仓库 ✅

| 仓库 | Stars | 增量叙事 | 是否采用 |
|---|---|---|---|
| microsoft/graphrag — https://github.com/microsoft/graphrag；论文 https://arxiv.org/abs/2404.16130（2024） | 36,696 | 把"检索片段"升级为"实体图 + 社区摘要"，回答**全局性/总结性**问题（"这批文档的主题是什么"） | **只讲不建**（成本，见 §7） |
| parthsarthi03/raptor — https://github.com/parthsarthi03/raptor；论文 https://arxiv.org/abs/2401.18059（2024） | 1,789 | 索引期递归聚类+摘要，形成多抽象层树；检索可在不同粒度间"折叠" | 采用为 R2 |
| Tongji-KGLLM/RAG-Survey — https://github.com/Tongji-KGLLM/RAG-Survey；论文 https://arxiv.org/abs/2312.10997（2023-12） | 2,138 | 提供 **Naive → Advanced → Modular** 三范式史观 + 检索/生成/增强三分法 | 采用为宏观叙事骨架 |
| explodinggradients/ragas — https://github.com/explodinggradients/ragas；文档 https://docs.ragas.io | 16,383 | 把 RAG 评测指标标准化（faithfulness / context precision / recall / response relevancy） | 采用为评测层 |
| Danielskry/Awesome-RAG — https://github.com/Danielskry/Awesome-RAG | 1,429 | 纯链接清单，无技术分期框架 | 不采用（🟡 不提供里程碑视角） |

---

## 5. 小型 RAG Demo 的评测实践

| 指标 | 测什么 | 成本 | 备注 |
|---|---|---|---|
| **Evidence recall@k**（自制） | gold evidence span 是否出现在 top-k | 零 LLM 成本、确定性 | 小项目最该先做；对 rung 的检索改动最敏感 |
| Context Recall | 参考答案中的信息能否在上下文中找到 | 1 次 judge | RAGAS 定义，https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/ |
| Context Precision | 相关 chunk 是否排在前面 | 1 次 judge | 对 rerank rung 敏感 |
| Faithfulness | 答案中的 claim 是否由 context 支持 | 1–N 次 judge | 对 Self-RAG/CRAG 敏感 |
| Response Relevancy | 答案是否切题 | 1 次 judge | **不测正确性**，别当 accuracy 用 |
| Answer correctness | 与 gold answer 一致 | 1 次 judge | 需人工抽检校准 |

**LLM-as-judge 的坑（✅ 有文献）**：位置偏差、冗长偏差、自我增强偏差、推理能力有限（Zheng et al., *Judging LLM-as-a-Judge*, https://arxiv.org/abs/2306.05685, 2023）。工程对策（🟡）：固定 judge 模型与 prompt 版本；答案顺序随机化或双序打分取平均；给 1–5 rubric 而不是 0/1；对 10–20% 样本人工复核；样本量 30–50 时只报区间与方向，不报小数点后两位。

**N/A 原则**：某 rung 不具备某能力（例如 R1 没有路由）时，表中写 `N/A` 并注明原因，**不要填 0**——否则跨 rung 的平均分会被系统性拉低，读者会误以为"能力退步"。

---

## 6. 推荐的 8 级阶梯

排序按"能力增量"而非严格年份；年份用于历史锚点。每级只允许新增**一个**机制。

| # | Rung（年份） | 代表引用 |  taxonomy | 相对上一级的 delta | 20 行实现长什么样（概念） |
|---|---|---|---|---|---|
| **R1** | Naive RAG（2020） | Lewis et al., https://arxiv.org/abs/2005.11401 | Generation（基线） | — 起点：chunk→embed→top-k→prompt | `split()`→`embed()`→余弦 top4→拼 prompt→`llm()`；LangGraph 两个节点一条边 |
| **R2** | RAPTOR（2024-01） | Sarthi et al., https://arxiv.org/abs/2401.18059 | Indexing | **索引期**：chunk 之上加"聚类→摘要"递归树，检索跨层 | 建树：`while len(nodes)>1: cluster→summarize`；检索：所有层扁平化后 top-k（collapsed tree） |
| **R3** | Multi-Query + RRF（2024-02） | Rackauckas, https://arxiv.org/abs/2402.03367 | Query Translation | **查询期**：1 问变 N 问，并行检索后按 `Σ1/(60+rank)` 融合 | `queries=llm(rewrite,n=4)`；`results=[retrieve(q) for q in queries]`；RRF 排序取前 4 |
| **R4** | HyDE（2022-12） | Gao et al., https://arxiv.org/abs/2212.10496 | Query Translation | **表示空间**：不 embed 问题，而 embed"假答案" | `fake=llm("写一段可能回答该问题的文字")`；`vec=mean(embed(fake_i))`；用 vec 检索 |
| **R5** | Rerank / Late Interaction（2020） | Khattab & Zaharia, https://arxiv.org/abs/2004.12832 | Retrieval | **精排**：先召回 20，再用 cross-encoder/ColBERT 打分取 4 | `cands=retrieve(k=20)`；`scores=cross_encoder(q,c)`；`top4=argsort(scores)[:4]` |
| **R6** | Adaptive RAG（2024-03） | Jeong et al., https://arxiv.org/abs/2403.14403 | Routing | **入口分支**：先判复杂度，决定不检索/单步/多步 | `route=llm_classify(q)`；LangGraph `add_conditional_edges(START, route, {...})`；三条入口路径 |
| **R7** | CRAG（2024-01） | Yan et al., https://arxiv.org/abs/2401.15884 | Iterative | **失败兜底**：检索评估器 + query 改写 + 外部检索回退 | `label=eval(retrieved)`；`if label!=correct: q2=rewrite(q); docs+=search(q2)`；再 generate |
| **R8** | Self-RAG（2023-10） | Asai et al., https://arxiv.org/abs/2310.11511 | Iterative | **生成后自评**：逐文档相关性 +  groundedness + 有用性，三处可回环 | `docs=[d for d in docs if isrel(q,d)]`；`ans=gen()`；`if not issup(ans,docs): ans=gen()`；`if not isuse(q,ans): q=rewrite(q); goto retrieve` |

**为什么是这 8 个而不是别的**（🟡）：
- 每一级都有**原始论文**，可引用、可讲"为什么当时出现"。
- 每一级只动一个旋钮：索引结构（R2）、查询数量（R3）、查询表示（R4）、排序精度（R5）、控制流入口（R6）、失败处理（R7）、生成后校验（R8）。
- 合起来覆盖六格 taxonomy 中的五格；唯一没建的是 Query Structuring（R5b），因为中文 Wiki 没有 SQL/Cypher 数据源，强行做会变成玩具。
- 没有把 GraphRAG 放进阶梯：它不是"再改一个旋钮"，而是换索引范式（实体图+社区摘要），且成本高一个量级。

---

## 7. 不建的 rung：GraphRAG（成本理由）

GraphRAG 的索引期要对**每个 chunk 做实体/关系抽取**，再对实体图做社区发现并为每个社区生成摘要（✅ 论文与仓库均如此描述）。对中文 Wiki 这种几千到上万 chunk 的语料，这意味着数千次抽取调用 + 数百次摘要调用，**索引成本比 Naive RAG 高一个数量级以上**（🟡 量级估计，未实测），且每次改语料都要重建。教学项目里它更适合作为"讲得清、跑不起"的一页：给出论文的 global sensemaking 例子与成本公式，标注 **N/A（未实现：全局摘要型问题）**。

同理，**Query Structuring** 也标 N/A：没有结构化数据源，硬做只能演示 metadata filter，信息增量低。

---

## 8. 这个项目凭什么不一样

| 差异点 | 具体做法 | 对比现有项目 |
|---|---|---|
| **单一状态机** | 一个 `GraphState`、一个节点注册表、一份 `edges` 配置；rung 是配置开关，不是 8 个仓库/8 本 notebook | NirDiamant 是 42 个互不调用的 notebook；LangGraph 四个教程各自独立 |
| **成本是一等公民** | 每个节点记录 `prompt_tokens / completion_tokens / 估算 USD / 延迟`；每级输出"索引一次性成本 + 每千问成本 + p50 延迟" | 现有教程基本只报效果，不报钱 |
| **中文 Wiki 语料** | 全 rung 共用同一份中文语料与同一批 40 题；中文分词/BM25/实体别名问题显式讨论 | 多数 demo 用英文博客（Lilian Weng），中文检索的坑不可见 |
| **诚实的 N/A** | 每级附能力矩阵：路由 / 网搜回退 / 自评 / 多跳 / 全局摘要；不具备就写 N/A + 原因 | 常见做法是把缺失能力算 0 分，制造"越来越差"的假象 |
| **可复现评测** | 同一 judge 模型、同一 prompt 版本、同一 gold evidence；区分确定性指标与 LLM 指标 | RAGAS 教程通常只给单次分数，不给跨系统对比协议 |

**一句话定位**（🟡）：不是"又一个 RAG 技术合集"，而是 **"RAG 的能力增量实验台"**——每个 rung 回答三个问题：多了什么节点、贵了多少、还有什么做不到。

---

## 9. 参考链接（全部本次可访问）

- NirDiamant/RAG_Techniques — https://github.com/NirDiamant/RAG_Techniques
- langchain-ai/rag-from-scratch — https://github.com/langchain-ai/rag-from-scratch
- Lance Martin 视频列表 — https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x
- Agentic RAG 教程 — https://docs.langchain.com/oss/python/langgraph/agentic-rag
- Self-Reflective RAG 博客（CRAG/Self-RAG 拓扑）— https://blog.langchain.com/agentic-rag-with-langgraph/
- CRAG notebook — https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb
- Self-RAG notebook — https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb
- Adaptive RAG 教程 — https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph/02-structures/07-langgraph-adaptive-rag
- RAG Survey — https://arxiv.org/abs/2312.10997
- GraphRAG — https://github.com/microsoft/graphrag ｜ https://arxiv.org/abs/2404.16130
- RAPTOR — https://github.com/parthsarthi03/raptor ｜ https://arxiv.org/abs/2401.18059
- RAGAS 指标 — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- LLM-as-judge 偏差 — https://arxiv.org/abs/2306.05685
- 原始论文：RAG https://arxiv.org/abs/2005.11401 ｜ HyDE https://arxiv.org/abs/2212.10496 ｜ ColBERT https://arxiv.org/abs/2004.12832 ｜ RAG-Fusion https://arxiv.org/abs/2402.03367 ｜ Adaptive-RAG https://arxiv.org/abs/2403.14403 ｜ Self-RAG https://arxiv.org/abs/2310.11511 ｜ CRAG https://arxiv.org/abs/2401.15884
