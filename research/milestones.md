# RAG 里程碑调研简报

> 目标：为一个 200 篇中文维基、869 个 chunk、约 37.5 万 token 的小语料，用 **一套 LangGraph 状态机** 演示 RAG 发展史中的 6–8 个里程碑。
> 约束：DeepSeek V4.1 Flash（$0.15/M 输入、$0.60/M 输出），总 LLM 预算 **$10**；bge-m3 与 bge-reranker-v2-m3 在 SiliconFlow 免费；GraphRAG 因成本与复杂度排除。
> 调研日期：2026-11-20。文中所有论文结论均给出链接；所有金额均为**估算**，不是账单。

---

## 0. 结论先行

**推荐的 8 级阶梯**（每一级只引入一个新的“控制点”，其余全部复用）：

| 级 | 里程碑 | 新增的控制点 | 代表工作 | 索引成本 | 150 题查询成本 |
|---|---|---|---|---|---|
| L1 | Naive RAG（BM25 / dense 两种检索器） | 先检索，再生成 | Lewis 2020；Karpukhin 2020 | $0 | $0.16（两种） |
| L2 | Hybrid + RRF + 精排（含 Lost-in-the-Middle 诊断） | 把“排序”做好 | Cormack 2009；bge-reranker | $0 | $0.09 |
| L3 | 查询变换：HyDE + Multi-Query/RAG-Fusion | 改查询，而不是改索引 | Gao 2022；Rackauckas 2024 | $0 | $0.21（两种） |
| L4 | Contextual Retrieval | 在索引期为 chunk 补上下文 | Anthropic 2024 | $0.15–0.37 | $0.09 |
| L5 | RAPTOR | 给语料建层次摘要树 | Sarthi 2024 | $0.08 | $0.08 |
| L6 | Self-RAG + CRAG（+ 可选 FLARE） | 生成前后自检、按置信度纠错 | Asai 2023；Yan 2024；Jiang 2023 | $0 | $0.32（两种） |
| L7 | Adaptive-RAG | 按问题复杂度选策略 | Jeong 2024 | $0 | $0.11 |
| L8 | Agentic RAG / ReAct | 多轮“想—查—看”循环 | Yao 2022 | $0 | $0.30（上限 $0.45） |

**总成本估算：8 个变体约 $2.25，11 个变体全跑约 $2.58；按 30% 重跑余量，落地预算约 $2.9–3.4；即使把所有可选变体都跑一遍也不超过 $4。** $10 预算有充足余量（见 §6）。

**一句话的“历史主线”**：RAG 的演进不是模型越来越大，而是**在流水线的不同位置加控制点**——先有检索（L1），再修排序（L2），再修查询（L3），再修索引（L4），再修语料结构（L5），再修生成的可信度（L6），再修“用哪条路”的策略（L7），最后把整条流水线交给 agent（L8）。

**已被排除/降级的候选**：

- **GraphRAG**：EXCLUDE。不是单纯因为 $0.55–1.0 的索引费，而是 200 篇文档只能长出很少的社区，全局题收益不稳定，且实体抽取/消歧/JSON 解析的工程量是 L5 的 5–10 倍；RAPTOR 已用 1/10 成本覆盖同一类“全局题”。
- **Late chunking**：EXCLUDE（在本题的技术栈下）。真正的 late chunking 需要 token 级隐状态，SiliconFlow 的 embedding 接口只返回池化后的向量；自托管 bge-m3 才能做，收益又和 L4 重叠。
- **Speculative RAG**：EXCLUDE。需要第二个小模型做 drafter，优化的是延迟而不是质量；869 个 chunk 的检索延迟根本不是瓶颈。
- **FLARE**：OPTIONAL。机制很干净，DeepSeek API 也支持 `logprobs`，但和 L6/L8 的循环高度重叠，建议作为 L8 的一个消融开关，而不是独立一级。
- **RAGAS**：不是里程碑，是**评测工具**，应当贯穿所有级别使用（见 §6.4）。

---

## 1. 事实基线（已核验）

### 1.1 语料

读了本地 `rag-ladder/data/manifest.json` 与 `manifest-chunks.json`，实际数字与任务描述一致，并补充几个对实现有用的细节：

| 项 | 值 | 来源 |
|---|---|---|
| 文档数 | 200 | `manifest.json` |
| chunk 数 | 869 | `manifest-chunks.json` |
| 切分 | 512 token / overlap 64，`cl100k_base`，中文标点递归切分 | 同上 |
| 总 token | **375,499** | 同上 |
| 每 chunk token | min 51 / median **478** / max 512 | 同上 |
| 每文档 chunk 数 | min 2 / median 4 / max 10 | 同上 |
| 主题 | person / geo / tech / hist / art 各 40 篇 | 同上 |

> 注意：869 × 512 = 445k 是上界；实际 375k，因为很多 chunk 没填满。成本估算一律用 **375k**，不要用 445k。

**实际字段与任务描述略有出入，写代码时以文件为准：**

- `corpus.jsonl`：`doc_id, title, theme, text, chars, url, license`（是 `url`，不是 `source_url`）。
- `chunks.jsonl`：`chunk_id, doc_id, title, theme, seq, text, start, end, tokens, chars`。`chunk_id` 形如 `person-744074a4bc14901f:000`，**必须全程冻结**，否则跨里程碑不可比。

### 1.2 评测集（重要偏差）

任务描述说“150 题，5 类：single / entity / multi_hop / global / no_answer”。**实际文件目前只有 30 题**，类型名也不同：

```
rows 30
Counter({'single': 12, 'entity': 6, 'multi': 6, 'global': 3, 'none': 3})
abstain 3
keys ['evidence','gold_answer','gold_doc_ids','qid','question','should_abstain','theme','type']
```

- 类型是 `multi` 与 `none`，不是 `multi_hop` 与 `no_answer`；没有 `gold_chunk_ids`，只有 `gold_doc_ids`。
- 因此：**先按 30 题跑通并出第一版数字，再补到 150 题**。§6.1 的成本表按 150 题给，§6.2 另给 30 题试点成本；150 题按 5 类各 30 题规划。
- `gold_chunk_ids` 可以由 `chunk_id` 前缀（`doc_id:`）自动展开，不需要重新标注；这样 `evidence_recall` 就能对 chunk 级计算。

### 1.3 价格与模型能力（按项目给定，未重新询价）

| 项 | 值 | 说明 |
|---|---|---|
| DeepSeek V4.1 Flash 输入 | $0.15 / 1M tokens | 项目给定 |
| DeepSeek V4.1 Flash 输出 | $0.60 / 1M tokens | 项目给定 |
| 缓存读（若启用） | $0.003 / 1M tokens | 来自项目此前的实测记录，约为输入价的 1/50 |
| bge-m3 | 免费（SiliconFlow） | 1024 维，**8192 token** 上下文 |
| bge-reranker-v2-m3 | 免费（SiliconFlow） | 8192 token 上下文 |
| DeepSeek `logprobs` | 支持，`top_logprobs` 0–20 | 这决定 FLARE 是否可做 |

参考：bge-m3 规格（1024 维 / 8192 token）见 Hugging Face 模型卡；DeepSeek 的 `logprobs` 参数见官方 Chat Completions 文档。

---

## 2. 成本模型（估算）

### 2.1 Token 假设

| 项 | 取值 | 依据 |
|---|---|---|
| 每个送入上下文的 chunk | 478 token | 语料中位数（实测） |
| system + 指令 | 250 token | 估算 |
| 问题本身 | 40 token | 估算 |
| 答案输出 | 220 token | 中文 150–250 字（估算） |
| 默认上下文 | 5 个 chunk，约 2,680 输入 token | 与 `MAX_CONTEXT_TOKENS=4000` 的旧纪律一致 |
| 精排后上下文 | 6 个 chunk，约 3,160 输入 token | 估算 |

**公式**：

```
cost = in_tokens × 0.15/1e6 + out_tokens × 0.60/1e6
```

### 2.2 单次生成的成本锚点

```
5 chunks:  2,680 × 1.5e-7 + 220 × 6e-7 = 0.000402 + 0.000132 = $0.000534
6 chunks:  3,160 × 1.5e-7 + 220 × 6e-7 = 0.000474 + 0.000132 = $0.000606
```

也就是说，**一道题的一次“检索 + 生成”大约 $0.0005–0.0006**；150 题约 **$0.08–0.09**。这就是为什么 $10 预算对这个规模非常宽裕——真正的成本在索引期的 LLM 调用和多次重跑，而不是答题。

### 2.3 估算的误差来源（必须在报告里承认）

1. 中文 token 数随 prompt 模板波动，±20% 很正常。
2. 失败重试、JSON 解析失败、LLM 输出超长，都会让实际账单高于本文估算；本文在总表里留了 30% 余量。
3. DeepSeek 的实际计费以账单为准；本文只做量级判断。
4. 若启用 prompt caching（同一文档重复出现在多个 chunk 的 contextual 提示中），L4 的索引成本可能降到 $0.05 以下；本文**没有**把缓存收益算进去，属于保守估算。

---

## 3. 候选里程碑逐条

每条的格式：**机制 → 解决的问题 → 在本语料上的可行性 → 成本 → 结论**。
统一标注：`[论文事实]` 表示来自论文/博客；`[估算]` 表示本文按 §2 模型推算。

**候选总表（先看结论）**

| # | 候选 | 索引 LLM | 150 题 | 结论 |
|---|---|---:|---:|---|
| C1 | Naive RAG | $0 | $0.08 | **INCLUDE（L1）** |
| C2 | DPR / dense retrieval | $0 | $0.08 | **INCLUDE（并入 L1）** |
| C3 | Hybrid + RRF + rerank | $0 | $0.09（+LITM 诊断 $0.05） | **INCLUDE（L2）** |
| C4 | HyDE | $0 | $0.11 | **INCLUDE（L3a）** |
| C5 | RAG-Fusion | $0 | $0.10 | **INCLUDE（L3b）** |
| C6 | Contextual Retrieval | $0.15–0.37 | $0.09 | **INCLUDE（L4）** |
| C7 | FLARE | $0 | $0.25 | OPTIONAL |
| C8 | Self-RAG | $0 | $0.23 | **INCLUDE（L6a）** |
| C9 | CRAG | $0 | $0.09 | **INCLUDE（L6b）** |
| C10 | RAPTOR | $0.08 | $0.08 | **INCLUDE（L5）** |
| C11 | Adaptive-RAG | $0 | $0.11 | **INCLUDE（L7）** |
| C12 | Agentic RAG / ReAct | $0 | $0.30（上限 $0.45） | **INCLUDE（L8）** |
| C13 | GraphRAG | $0.55–1.0 | $0.50–1.50 | **EXCLUDE** |
| C14 | Late chunking | $0（需自托管） | $0 | **EXCLUDE** |
| C15 | Speculative RAG | $0 | $0.24 | **EXCLUDE** |
| C16 | RAGAS | $0 | $0.08/级 | **TOOL** |
| C17 | Lost in the Middle | $0 | $0.05（30 题诊断） | **DIAGNOSTIC** |

---

### C1. Naive RAG / 原始 RAG

- **代表工作**：Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020，[arXiv:2005.11401](https://arxiv.org/abs/2005.11401)。
- **核心机制**：把预训练 seq2seq（BART）与一个可微检索器（DPR）组合，把检索到的文档当作**隐变量**，用 top-k 近似做边缘化；有 RAG-Sequence（整段用同一文档）与 RAG-Token（每个 token 可换文档）两种形式，并**联合微调**检索器 query encoder 与生成器。
- **解决的问题**：纯参数化模型无法更新知识、无法给出出处；当时的开放域 QA 需要“检索 + 生成”的通用配方。
- **重要的历史辨析**：原论文是**端到端训练**的 RAG，不是今天说的“向量检索 + 拼 prompt”管道。“Naive RAG”这个词是后来的综述（Gao et al., 2023，[arXiv:2312.10997](https://arxiv.org/abs/2312.10997)）对“indexing → retrieval → generation”三段式管道的追认。做“历史”项目时，这个区别本身就是一个很好的讲解点。
- **可行性**：高。我们不能在本机微调 BART，但可以**演示机制**：固定检索器 + 固定生成器，一次性 top-k 检索，拼接上下文生成。这是“概念复现”，不是“论文复现”，必须在 README 里写清楚。
- **成本** `[估算]`：索引 $0（embedding 免费）；150 题 × $0.000534 ≈ **$0.080**（BM25 与 dense 各跑一遍则 $0.16）。
- **结论**：**INCLUDE（L1，必需基线）**。没有它，后面所有“提升”都没有分母。

---

### C2. Dense Passage Retrieval（DPR）

- **代表工作**：Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020，[arXiv:2004.04906](https://arxiv.org/abs/2004.04906)；[ACL Anthology](https://aclanthology.org/2020.emnlp-main.550/)。
- **核心机制**：双塔（dual-encoder）分别编码 question 与 passage，用点积做相似度，用 in-batch negatives + BM25 硬负例训练；索引离线用 FAISS。top-20 检索准确率比 BM25 绝对提升 9–19 个百分点。
- **解决的问题**：稀疏检索（TF-IDF/BM25）对同义改写、语义匹配无能为力；DPR 证明“只用稠密向量”也能做 passage retrieval。
- **可行性**：**机制可演示，训练不可复现**。200 篇文档、30 道题不足以训练 DPR；直接用 bge-m3 作为双塔稠密检索器，属于“同一机制的现代替代”。这正好可以讲清楚 DPR 的历史地位：它奠定了 bi-encoder + 向量索引的范式，而 bge-m3 是这条路线上的现代产物。
- **成本** `[估算]`：索引 $0（bge-m3 免费）；150 题 ≈ **$0.080**。
- **结论**：**INCLUDE，但并入 L1 作为 dense 分支**，不单独占一级。这样 L1 天然有 BM25 vs dense 的对照，成本只多 $0.08。

---

### C3. Hybrid Search + Reciprocal Rank Fusion + Cross-encoder Rerank

- **代表工作**：Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009，[DOI:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)，[PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)；BM25 综述见 Robertson & Zaragoza, 2009，[DOI:10.1561/1500000019](https://doi.org/10.1561/1500000019)。
- **核心机制**：把多个检索系统的排名用 `RRF(d) = Σ 1/(k + rank_i(d))` 融合，论文取 **k = 60**；不依赖各系统的原始分数，只依赖名次。工程上再接一个 cross-encoder 精排（bge-reranker-v2-m3）把候选从 50 压到 5–8。
- **解决的问题**：BM25 擅长专名/编号/精确串，dense 擅长语义改写；单用任何一路都会漏。RRF 用“名次共识”把两路合并，reranker 再用 query-document 交互把真正相关的排到前面。Anthropic 的 Contextual Retrieval 博客也报告：embedding + BM25 优于单用 embedding，再加 rerank 后检索失败率从 5.7% 降到 1.9%（相对降 67%）。
- **可行性**：很高。中文 BM25 需要 jieba 分词，`bm25s` 足够；bge-m3 dense 已可用；bge-reranker 免费。869 个 chunk 的 BM25 索引是毫秒级。
- **成本** `[估算]`：索引 $0；150 题 ≈ **$0.091**（只是生成成本，融合与精排无 LLM 调用）。
- **附加诊断（Lost in the Middle）**：Liu et al., 2023，[arXiv:2307.03172](https://arxiv.org/abs/2307.03172) 证明相关信息放在长上下文中间时性能显著下降（论文中 GPT-3.5-Turbo 的性能下降可超过 20%，呈 U 形，最坏时甚至低于不给文档的 closed-book 设置）。这不是一个独立方法，而是 L2 的**零成本消融**：把同一批 6 个 chunk 按“最相关在前 / 最相关在中间 / 最相关在最后”三种顺序各跑一遍，在 30 题子集上比较即可。成本约 **$0.05**（30 题 × 3 种顺序 × $0.0006）。
- **结论**：**INCLUDE（L2）**。这是“工程 RAG”性价比最高的一级，也是全项目最可能看到最大单次增益的一级。

---

### C4. HyDE（Hypothetical Document Embeddings）

- **代表工作**：Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, 2022，[arXiv:2212.10496](https://arxiv.org/abs/2212.10496)；ACL 2023，[ACL Anthology](https://aclanthology.org/2023.acl-long.99/)。
- **核心机制**：先让 LLM 针对问题“写一篇假想答案文档”，再用**文档编码器**把它编码成向量去检索真实文档；生成文档里的事实错误被编码器的稠密瓶颈“过滤”掉，留下的是相关性模式。论文还给出一个变体：把假想文档向量与原始 query 向量做平均。
- **解决的问题**：零样本 dense retrieval 没有 relevance label，query 与 document 在向量空间里不对称（短 query vs 长文档）。HyDE 把 query 变成“像文档的东西”，把 query-document 匹配转成 document-document 匹配。
- **可行性**：高。中文维基上让 DeepSeek 写 150–250 字假想段落即可；bge-m3 同时编码 query 与 document，无需训练。风险是模型可能写出具体但错误的年份/人名，不过在 200 篇小语料上，稠密瓶颈通常仍能落到正确主题附近。
- **成本** `[估算]`：索引 $0；每题多一次假想文档生成（约 280 in / 220 out ≈ $0.000176），150 题 ≈ **$0.107**。
- **结论**：**INCLUDE，作为 L3 的变体 A**。它代表“query → document 空间”的一次重要观念转变。

---

### C5. RAG-Fusion / Multi-Query

- **代表工作**：Rackauckas, *RAG-Fusion: a New Take on Retrieval-Augmented Generation*, IJNLC 2024，[arXiv:2402.03367](https://arxiv.org/abs/2402.03367)。
- **核心机制**：LLM 把原问题改写成 4 个不同视角的查询，每个查询独立检索，得到多个排名列表，再用 RRF（k=60）融合；本质是“多查询 + 排名融合”。
- **解决的问题**：单一 query 的措辞会限制召回；用户问“A 对 B 有什么影响”时，可能漏掉只讲“B 受到 A 影响”的文档。多视角改写提高召回，RRF 抑制单个坏改写带来的噪声。
- **可行性**：很高。一次调用生成 4 条改写即可；RRF 是纯算法。注意论文的评测是人工小样本，结论应表述为“作者报告更全面”，而不是“已证明普遍更好”。另一篇独立评测（RAGElo，[arXiv:2406.14783](https://arxiv.org/abs/2406.14783)）发现 RAG-Fusion 提升“完整性”但可能损害“精确性”，这个 trade-off 值得在项目里展示。
- **成本** `[估算]`：索引 $0；每题改写约 290 in / 180 out ≈ $0.000152，加生成 ≈ **$0.000686**，150 题 ≈ **$0.103**。
- **结论**：**INCLUDE，作为 L3 的变体 B**。和 HyDE 放在同一级，是因为它们回答同一个问题：“在检索之前，我们还能对 query 做什么？”

---

### C6. Contextual Retrieval

- **代表工作**：Anthropic, *Introducing Contextual Retrieval*, 2024-09-19，[博客](https://www.anthropic.com/research/contextual-retrieval)；[实验附录 PDF](https://www-cdn.anthropic.com/5722e7658c9302d8b97a3238de1bb8e6afdf04b9.pdf)。
- **核心机制**：在索引前，让 LLM 为每个 chunk 生成 50–100 token 的“它在整篇文档中处于什么位置”的说明，拼在 chunk 前面，然后**用带上下文的文本**分别建 embedding 索引和 BM25 索引（Contextual Embeddings + Contextual BM25）。原始 chunk 本身不变，`chunk_id` 不变。
- **解决的问题**：切分破坏了上下文。一个 chunk 里写“该公司营收增长 3%”，但没说是哪家公司、哪个季度；嵌入和 BM25 都会失效。Anthropic 报告：Contextual Embeddings 把 top-20 检索失败率从 5.7% 降到 3.7%（相对降 35%）；再加 Contextual BM25 降到 2.9%（降 49%）；再加 rerank 降到 1.9%（降 67%）。
- **可行性**：高，但要控制成本。200 篇文档平均 1,877 token，完全可以**按文档批量生成**：一次调用喂整篇文档 + 该文档的所有 chunk，让模型输出每段的上下文说明。中文维基的 chunk 经常以“他/该市/该公司”开头，这个补全非常对症。
- **成本** `[估算]`：
  - 按文档批量：输入 ≈ 375k（文档）+ 375k（chunk）+ 50k（指令）= 800k → $0.12；输出 869 × 60 ≈ 52k → $0.031；合计 **≈ $0.15**。
  - 按 chunk 逐条（每条都带整篇文档）：输入 ≈ 2.26M → $0.34；输出同上 → **≈ $0.37**。
  - 建议用前者，预算按 **$0.25** 记。
  - 查询期与 L2 相同：150 题 ≈ **$0.091**（6 个 chunk）。
- **结论**：**INCLUDE（L4）**。这是“索引期智能”的代表，和 L3 的“查询期智能”形成漂亮对称；也是唯一需要认真记账的索引步骤（除 RAPTOR 外）。

---

### C7. FLARE（Forward-Looking Active REtrieval）

- **代表工作**：Jiang et al., *Active Retrieval Augmented Generation*, EMNLP 2023，[arXiv:2305.06983](https://arxiv.org/abs/2305.06983)；[ACL Anthology](https://aclanthology.org/2023.emnlp-main.495/)。
- **核心机制**：不要“先检索一次再生成”，而是**边生成边决定**。`FLARE_direct` 先让模型生成下一句（不接检索），如果这句话里有 token 概率低于阈值 θ，就把这句话（或掩码后的版本 / 生成的问题）当查询去检索，再重新生成这句；`FLARE_instruct` 则让模型自己输出 `[Search(query)]`。
- **解决的问题**：长文本生成中，信息需求是逐步出现的；一次性检索无法预知后面要说什么。FLARE 让检索时机和检索 query 都由生成过程决定。
- **可行性**：**可做，且比想象中顺**。DeepSeek 的 Chat Completions 支持 `logprobs` 与 `top_logprobs`，所以 `FLARE_direct` 的“低置信 token”信号是真实可得的，不需要用 prompt 假装。风险是循环次数与成本；必须硬性限制 `max_iterations=3`。
- **成本** `[估算]`：每题约 3 次“生成句子 + 可能重生成” ≈ 2,000–3,000 in / 400–600 out，加最终答案，约 **$0.0015–0.0020/题**，150 题 ≈ **$0.25**。
- **结论**：**OPTIONAL**。机制值得讲，但它和 L6（Self-RAG 的“要不要检索”）、L8（ReAct 的多轮）重叠；如果做成独立一级，学生会觉得“又是循环”。建议作为 L8 的一个开关（`active_retrieval=True`）或 L6 的附加变体。

---

### C8. Self-RAG

- **代表工作**：Asai et al., *Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection*, ICLR 2024，[arXiv:2310.11511](https://arxiv.org/abs/2310.11511)；[项目页](https://selfrag.github.io/)。
- **核心机制**：训练一个 LM，让它在输出文本的同时生成四类**反思 token**：`Retrieve`（yes/no/continue，决定要不要检索）、`IsRel`（检索到的段落相关/不相关）、`IsSup`（生成内容被段落支持的程度：full/partial/none）、`IsUse`（整体有用性 1–5）。推理时用 segment-level beam search，按这些 token 的概率选最好的续写。
- **解决的问题**：传统 RAG 不管需不需要都固定检索 top-k，也不检查生成是否真的被证据支持。Self-RAG 把“何时检索、检索到的是否有用、生成是否有据”变成模型自己的决策。
- **可行性**：**机制可演示，训练不可复现**。原论文要构造 critic 数据（用 GPT-4 标注）并端到端训练一个 LLaMA-2；我们只能用 DeepSeek 的 prompt 来模拟四个判断：一次“是否需要检索”、一次批量“每个 chunk 是否相关”、生成后一次“是否被支持”。这是 **Self-RAG-style**，不是 Self-RAG 本体，必须在报告里明确。
- **成本** `[估算]`：每次查询约 4 次调用（决策 + 相关性批量 + 生成 + 支持性检查），约 7,800 in / 370 out ≈ $0.00139；按 25% 触发一次重试（+$0.0005 × 0.25），约 **$0.00152/题**，150 题 ≈ **$0.228**。
- **结论**：**INCLUDE（L6 变体 A）**。它是“生成后验证”的代表，也是 `no_answer` 题最需要的机制。

---

### C9. CRAG（Corrective RAG）

- **代表工作**：Yan et al., *Corrective Retrieval Augmented Generation*, 2024，[arXiv:2401.15884](https://arxiv.org/abs/2401.15884)。
- **核心机制**：用一个轻量检索评估器（论文是 0.77B 的 T5-large）给每个“问题-文档”对打相关性分，聚合成三档置信度：**Correct / Ambiguous / Incorrect**。Correct → 对文档做 knowledge refinement（分解、过滤、重组）；Incorrect → 丢弃，改用 web search；Ambiguous → 两者都做，再交给生成器。
- **解决的问题**：检索可能整体错掉，但传统 RAG 仍会把错误文档塞进 prompt，导致“自信地胡说”。CRAG 在生成前先判断“这批检索到底靠不靠谱”，并给出纠错动作。
- **可行性**：**高，而且比原论文更适合本项目**。不需要训练 0.77B 评估器：直接用 bge-reranker 的分数做置信度（免费），或用一次 DeepSeek 三分类调用。原论文的 web search 分支在离线评测里不可复现，应改为“查询改写 + 放宽 top-k + 换检索器（BM25↔dense）”，并在报告中注明这是**离线替身**。
- **成本** `[估算]`：精排分数免费；生成 $0.000534；30% 的题触发一次改写（约 $0.00015）；合计 **$0.00058/题**，150 题 ≈ **$0.087**。
- **结论**：**INCLUDE（L6 变体 B）**。它比 Self-RAG 便宜、更容易实现，适合作为 L6 的第一个版本；Self-RAG 作为“更细粒度”的第二个版本。

---

### C10. RAPTOR

- **代表工作**：Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*, ICLR 2024，[arXiv:2401.18059](https://arxiv.org/abs/2401.18059)。
- **核心机制**：把 chunk 当叶子，递归地做“嵌入 → 聚类（GMM + UMAP，软聚类）→ LLM 摘要 → 再嵌入”，形成一棵多层次摘要树；查询时用 **collapsed tree**（所有层一起算相似度）或 tree traversal 检索。论文报告在 QuALITY 上配合 GPT-4 比此前最好结果绝对提升 20%。
- **解决的问题**：只检索短 chunk 无法回答“整篇/整个语料在讲什么”的全局问题；树提供了不同抽象层级的节点，让查询自己选择粒度。
- **可行性**：高。869 个叶子，按每簇 8–12 个摘要，大约得到 87 个一级节点、9 个二级节点、1 个根，总共约 100 次摘要调用。**注意**：RAPTOR 原论文是在单篇长文档内部建树；我们是在**整个语料**上建树（跨文档聚类），这更接近“轻量 GraphRAG”，也是对 200 篇小语料的合理改编。要在报告里说明这个 deviation。
- **成本** `[估算]`：一级 87 × (10×478 + 250) ≈ 440k in；二三级约 20k in；输出约 100 × 180 ≈ 18k。合计 **≈ $0.08**。查询期 8 个混合层级节点（平均 300 token）≈ **$0.080/150 题**。
- **结论**：**INCLUDE（L5）**。它是“全局题”最便宜的答案，也是“层次化索引”这一课的最佳载体。

---

### C11. Adaptive-RAG

- **代表工作**：Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*, NAACL 2024，[arXiv:2403.14403](https://arxiv.org/abs/2403.14403)；[ACL Anthology](https://aclanthology.org/2024.naacl-long.389/)。
- **核心机制**：训练一个小分类器，把问题分成 A（不需要检索）/ B（单步检索）/ C（多步检索）三档，再路由到对应策略。分类器的训练标签是自动构造的：先看哪个策略能答对，再按“简单优先”打标。
- **解决的问题**：对所有问题都用最重的多步策略太贵、太慢；都用最简单的单步策略又答不了复杂题。Adaptive-RAG 用“问题复杂度”做成本-质量的折中。
- **可行性**：**路由可做，分类器不可训**。30/150 道题不足以训练 T5-large 分类器；用一次零样本 LLM 调用做 A/B/C 路由，或直接用规则（`qtype` 字段 + 问句长度）做先验。这是 **Adaptive-RAG-style**，必须在报告中注明。
- **成本** `[估算]`：路由约 290 in / 5 out ≈ $0.000047；按 15% 直答、65% 单步、20% 多步混合，期望 **$0.00071/题**，150 题 ≈ **$0.107**。
- **结论**：**INCLUDE（L7）**。它是整条阶梯的“策略层”，让前面 6 级从“一堆可选管道”变成一个会选路的系统。

---

### C12. Agentic RAG / ReAct

- **代表工作**：Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023，[arXiv:2210.03629](https://arxiv.org/abs/2210.03629)；综述见 Singh et al., *Agentic Retrieval-Augmented Generation: A Survey*, 2025，[arXiv:2501.09136](https://arxiv.org/abs/2501.09136)。
- **核心机制**：把“思考（thought）—行动（action）—观察（observation）”交错起来。模型先写下要做什么，再调用工具（`search(query)` / `lookup` / `finish`），看到结果后继续想，直到给出答案。ReAct 在 HotpotQA/FEVER 上通过“检索 + 推理交错”减少幻觉；在 ALFWorld/WebShop 上比模仿学习/强化学习基线成功率绝对提升 34%/10%（论文口径，对应 Table 3/4）。
- **解决的问题**：多跳问题需要“先查 A，再从 A 里得到线索去查 B”；固定的一次性检索做不到。ReAct 把检索变成 agent 的动作，让模型自己决定查什么、查几次、什么时候停。
- **可行性**：高，但要设硬上限。工具只需要一个：`search(query) -> top-5 chunks + scores`；`max_steps=5`，总上下文不超过 4,000 token；终止条件为 `FINAL:` 或步数上限。中文维基的实体名适合作为查询词。
- **成本** `[估算]`：按平均 4 步（提示随历史增长）约 7,400 in / 480 out，加最终综合 3,000 in / 250 out，约 **$0.0020/题**；150 题 ≈ **$0.300**；若平均 6 步，上限约 **$0.45**。
- **结论**：**INCLUDE（L8）**。作为“终点”很合适：它把前面所有能力（检索、精排、改写、自检）都变成可选工具，也最能让读者直观看到“agent 不是新检索器，而是新的控制流”。

---

### C13. GraphRAG（排除项，但要把理由讲清楚）

- **代表工作**：Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, 2024，[arXiv:2404.16130](https://arxiv.org/abs/2404.16130)；[官方 indexing methods](https://microsoft.github.io/graphrag/index/methods/)。
- **核心机制**：用 LLM 从每个 chunk 抽实体与关系 → 合并成实体图 → Leiden 层次社区检测 → 为每个社区预生成“社区报告” → 查询时 Local search（实体邻域）或 Global search（社区报告 map-reduce）。
- **解决的问题**：全局性问题（“这批文档的主要主题是什么？”）本质是 query-focused summarization，向量 top-k 检索没有对应机制；图 + 社区摘要提供了“先总结再回答”的结构。
- **可行性/成本** `[估算]`：按 869 个 chunk 抽实体/关系，输入约 635k、输出约 217k，基础成本约 $0.25–0.30；加实体摘要、社区报告、JSON 解析重试（按 2× 计），约 **$0.55–1.0**，与任务给出的区间一致。查询期 global search 要把几十份社区报告放进上下文，约 **$0.003–0.01/题**，150 题再加 $0.5–1.5。
- **为什么仍然 EXCLUDE**（成本不是唯一原因）：
  1. **图太小**：200 篇文档的实体图可能只有几百到几千个节点，Leiden 只能切出很少的社区，社区报告会很薄，全局题的表现不稳定。
  2. **工程量**：实体抽取、指代消歧、别名合并、社区层级选择、map-reduce 提示，任何一环都能吃掉几天；这不是“加一个节点”，而是“加一个索引系统”。
  3. **被 RAPTOR 覆盖**：对“全局/主题”类问题，L5 用 $0.08 就能给出层次摘要；两者在 200 篇规模上的差距远小于实现成本差距。
  4. **可讲清楚**：在报告里用一页解释“GraphRAG 把索引变成图，代价是索引期 LLM 调用和实体解析”，比硬做一遍更有教学价值。
- **补充**：Microsoft 后来提出 LazyGraphRAG（[博客](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)），把索引成本降到与向量 RAG 相同、约为 full GraphRAG 的 0.1%，查询成本也大幅下降。这说明 GraphRAG 的问题正在被工程化解决，但在本项目的时间/预算下仍不值得。
- **结论**：**EXCLUDE**。

---

### C14. Late Chunking / Long-context Embedding

- **代表工作**：Günther et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*, 2024，[arXiv:2409.04701](https://arxiv.org/abs/2409.04701)；Jina 博客，[链接](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)。
- **核心机制**：先让长上下文 embedding 模型编码**整篇文档的所有 token**，得到 token 级隐状态；再按 chunk 边界对 token 向量做 mean pooling，得到“看过全文”的 chunk 向量。与“先切再嵌入”相反，所以叫 late。
- **解决的问题**：和 Contextual Retrieval 同一个病：chunk 失去上下文。区别是 late chunking 不改文本、不加 LLM 生成，纯粹在 embedding 层解决。
- **可行性**：**在本题的技术栈下不可行**。bge-m3 虽然支持 8,192 token，但 SiliconFlow 的 embeddings API 只返回池化后的 1024 维向量，不暴露 token 级隐状态；要真正做 late chunking，必须自托管 bge-m3（568M 参数）并跑 8k 序列。200 篇文档在 CPU 上可能尚可，但会引入一整套部署/显存/批处理工程，收益与 L4 高度重叠。
- **成本** `[估算]`：若自托管，LLM $0；embedding 电费/时间另计。查询期 $0（与 L1 相同）。
- **结论**：**EXCLUDE**。在报告里用一段话说明“late chunking 与 contextual retrieval 解决同一个问题，一个在 embedding 层、一个在文本层；我们选后者，因为免费 API 支持”。这比硬做更有历史纵深感。

---

### C15. Speculative RAG

- **代表工作**：Wang et al., *Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting*, 2024，[arXiv:2407.08223](https://arxiv.org/abs/2407.08223)。
- **核心机制**：用一个小而专的 drafter 模型，对检索文档的多个不同子集并行生成多份“答案草稿 + 理由”；再用一个大而通用的 verifier 模型，一次前向给每份草稿打分，选最高分。论文报告在 PubHealth 上准确率提升最多 12.97%、延迟降低 50.83%。
- **解决的问题**：长上下文里信息分散、位置偏差；单次生成容易被某一段带偏。多草稿 + 验证可以在不增加 verifier 上下文的情况下提升鲁棒性。
- **可行性**：机制可模拟（同一模型生成 4 份草稿，再让同一模型当 verifier），但这样就失去了“小 drafter 提速”的核心卖点；要真正复现需要第二个模型。869 个 chunk 的场景里，延迟不是瓶颈，质量增益也不稳定。
- **成本** `[估算]`：4 份草稿 + 1 次验证，约 7,200 in / 850 out ≈ **$0.0016/题**，150 题 ≈ **$0.24**。
- **结论**：**EXCLUDE**（低优先级 OPTIONAL）。不是坏方法，而是它解决的是“大模型 RAG 的延迟”，不是本项目要讲的历史节点。

---

### C16. RAGAS（评测框架，不是里程碑）

- **代表工作**：Es et al., *RAGAs: Automated Evaluation of Retrieval Augmented Generation*, EACL 2024 Demo，[arXiv:2309.15217](https://arxiv.org/abs/2309.15217)；[ACL Anthology](https://aclanthology.org/2024.eacl-demo.16/)；[Context Recall 文档](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)。
- **核心机制**：用 LLM 做三类无参考指标：faithfulness（答案的每个 claim 是否能被上下文推出）、answer relevance（由答案反推问题，再算与原问题的相似度）、context relevance（上下文里有多少句是真正需要的）；有标注时再加 context recall/precision。
- **解决的问题**：RAG 的评测不能只看最终答案，还要看“检索对没对、生成有没有根据”。RAGAS 把这些维度拆开，且不需要人工逐条标注。
- **可行性**：高。但要**只把它当工具**：本项目有 `gold_answer` 与 `gold_doc_ids`，可以先用确定性指标（EM/F1、evidence_recall、abstain accuracy），再用 RAGAS 的 faithfulness 做补充。全量 150 题 × 每级一次 judge ≈ **$0.081/级**；若只抽样 30 题，≈ **$0.016/级**。
- **结论**：**TOOL（使用，不作为一级）**。建议在报告里单独写“评测协议”，而不是把它混进“RAG 发展史”。

---

### C17. Lost in the Middle（诊断，不是里程碑）

- **代表工作**：Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024，[arXiv:2307.03172](https://arxiv.org/abs/2307.03172)；[ACL Anthology](https://aclanthology.org/2024.tacl-1.9/)。
- **核心机制**：控制“答案所在文档”在输入上下文中的位置，发现性能呈 U 形：开头和结尾最好，中间显著变差；即使模型支持更长上下文，也不代表它会均匀使用。
- **解决的问题**：提醒我们“检索对了”不等于“模型用上了”；排序和位置本身是变量。
- **可行性**：零额外检索成本，只需改变拼接顺序重跑生成。建议在 L2 上做 30 题 × 3 种顺序的消融。
- **成本** `[估算]`：30 题 × 3 × $0.0006 ≈ **$0.054**（若只做 2 种顺序约 $0.036）。
- **结论**：**DIAGNOSTIC（并入 L2）**。

---

## 4. 推荐的 8 级阶梯与顺序

### 4.1 为什么是这个顺序

一条好的“历史阶梯”应当满足三个条件：**每一级只改一个变量**、**每一级都能被上一级的失败解释**、**每一级都能在同一套评测上留下痕迹**。

| 顺序 | 上一级的典型失败 | 这一级引入的修复 | 预期在哪些题型上见效 |
|---|---|---|---|
| L1 Naive RAG | — | 检索 + 生成 | 单跳事实 |
| L2 Hybrid + RRF + rerank | 专名/编号漏检；语义改写漏检；相关 chunk 排在后面 | 稀疏+稠密双路、名次融合、cross-encoder 精排 | entity、single |
| L3 HyDE + RAG-Fusion | query 太短、措辞单一，召回不到 | 查询侧扩展（假想文档 / 多视角改写） | single、multi |
| L4 Contextual Retrieval | chunk 缺主语/缺时间，索引本身信息不全 | 索引期补上下文（文本 + BM25 同步） | single、entity、multi |
| L5 RAPTOR | 只能看到局部 chunk，答不了“整体” | 层次摘要树，跨层检索 | global、multi |
| L6 Self-RAG + CRAG | 检索错了还硬答；生成了没根据的话 | 相关性判断、支持性判断、纠错/弃答 | no_answer、multi |
| L7 Adaptive-RAG | 所有题都走最重的路径，贵且慢；简单题被过度加工 | 按复杂度路由 | 全题型（成本维度） |
| L8 Agentic RAG | 多跳需要“查到线索再查” | ReAct 循环 + 工具 | multi、global |

这个顺序还有一个叙事上的好处：**L1–L2 是“检索工程”，L3–L4 是“查询 vs 索引”的对称，L5 是“语料结构”，L6–L8 是“控制流”**。讲的时候可以明确说：RAG 的历史不是一条直线，而是这四个战场的轮番推进。

### 4.2 如果只能做 6 级

保留 **L1、L2、L3、L4、L6、L8**，把 L5（RAPTOR）与 L7（Adaptive）放进“可选扩展”。理由：L5 只对 global 题有独特价值，L7 是策略层、可以最后补；而 L1/L2/L4/L6/L8 分别代表检索、排序、索引、验证、自治五个不可替代的观念。若时间允许，L5 应优先于 L7 补回，因为它是“全局题”的唯一低成本答案。

### 4.3 每一级“不能解决什么”（写进报告，避免过度承诺）

- L1 不能解决词汇不匹配，也不能解决多跳。
- L2 不能解决“索引里根本没有这条信息”。
- L3 不能解决 chunk 本身缺上下文；HyDE 还可能引入虚假细节。
- L4 不能解决跨文档的全局问题，且索引期有 LLM 成本。
- L5 的摘要有约 4% 的轻微幻觉（RAPTOR 论文自报），200 篇规模上更明显。
- L6 的评估器会误判，可能把正确答案弃掉；必须同时报“误弃率”。
- L7 的路由器在 30 题上无法训练，零样本路由会有错分。
- L8 的轨迹不稳定，必须硬性步数上限；单跳题可能因过度检索而变差。

---

## 5. 用一套 LangGraph 状态机装下 8 级

### 5.1 设计原则

不要写 8 个 `pipeline.py`。正确做法是：

1. **两个图**：一个**离线建索引图**（build graph），一个**在线查询图**（query graph）。索引成本只发生在 build graph 里。
2. **里程碑 = 查询图的一组配置 + 少量条件分支**，不是一套新代码。
3. **共享节点**：`retrieve`、`fuse`、`rerank`、`generate`、`critique`、`finalize` 对所有级别是同一份实现。这样指标才可比，bug 只修一次。
4. **差异只在三个地方**：`expand` 的模式（none/hyde/multi_query/decompose）、`retrieve` 的索引适配器（bm25/dense/contextual/raptor）、`route` 的策略。

### 5.2 状态定义（建议直接照抄）

```python
from typing import Annotated, Literal, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class Chunk(TypedDict):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str          # bm25 | dense | contextual | raptor

class RAGState(TypedDict, total=False):
    # ---- 输入 ----
    qid: str
    question: str
    qtype: str                  # single | entity | multi | global | none
    # ---- 控制 ----
    strategy: str               # L1_bm25 | L1_dense | L2_hybrid | L3_hyde | ...
    route: Literal["direct", "single", "multi", "agent"]
    expand_mode: Literal["none", "hyde", "multi_query", "decompose"]
    index: str                   # bm25 | dense | hybrid | contextual | raptor
    use_rerank: bool
    use_grade: bool
    use_critique: bool
    max_rounds: int
    round: int
    # ---- 检索 ----
    queries: list[str]
    candidates: Annotated[list[Chunk], merge_chunks]   # 并行分支写入，按 chunk_id 去重取 max score
    contexts: list[Chunk]
    confidence: float
    verdict: Literal["correct", "ambiguous", "incorrect"]
    # ---- 生成 ----
    draft: str
    answer: str
    citations: list[str]
    supported: Literal["full", "partial", "none"]
    # ---- 观测 ----
    trace: Annotated[list[dict], add]                  # 每个节点写一条事件
    usage: Annotated[dict, merge_usage]                # tokens / calls / latency
```

两个 reducer 是关键：

- `merge_chunks`：按 `chunk_id` 去重，保留最高分。RAG-Fusion 的多路检索、Adaptive 的多步子查询都会并发写 `candidates`，没有 reducer 会互相覆盖。
- `merge_usage`：累加 token 与调用数，用于成本表。

> 注意：`merge_chunks` / `merge_usage` 必须在 `class RAGState` **之前**定义——Python 的 `Annotated` 注解在类定义时就会求值。本文为了阅读顺序把函数放在了 §5.4，实际写代码时请上移。

### 5.3 节点清单

| 节点 | 职责 | 对应里程碑 | 是否调用 LLM |
|---|---|---|---|
| `prepare` | 规范化问题、读 `qtype`、初始化预算 | 全部 | 否 |
| `route` | 判断 A/B/C 或直接指定策略 | L7 | 是（1 次小调用） |
| `expand` | `hyde` 生成假想文档；`multi_query` 生成 4 条改写；`decompose` 拆子问题 | L3 | 是 |
| `retrieve` | 按 `state["index"]` 调 BM25 / dense / contextual / RAPTOR；返回带分数的候选 | L1–L5 | 否 |
| `fuse` | RRF(k=60) 合并多个排名列表 | L2/L3 | 否 |
| `rerank` | bge-reranker 对候选精排，截断到 6 个 | L2–L8 | 否（免费 API） |
| `grade` | 三档置信度（reranker 分数阈值，或 LLM 三分类） | L6/L7 | 可选 |
| `generate` | 用统一模板生成答案与引用 | 全部 | 是 |
| `critique` | `IsSup` / `IsUse`：答案是否被上下文支持 | L6 | 是 |
| `rewrite` | 根据失败原因改写查询或放宽检索 | L6/L8 | 是 |
| `agent` | ReAct 的一步：产出 thought + action | L8 | 是 |
| `tool_search` | 执行 `search(query)`，把 observation 写回 | L8 | 否 |
| `finalize` | 统一输出格式、弃答判定、写 trace | 全部 | 否 |

### 5.4 边与条件边

```
START → prepare → route
route ──direct──→ generate
      ──single──→ retrieve → fuse? → rerank → [grade?] → generate → [critique?] → finalize
      ──multi───→ expand → retrieve(×N) → fuse → rerank → generate
      ──agent───→ agent ⇄ tool_search → finalize
```

具体条件边（LangGraph 的 `add_conditional_edges` 接收当前 state，返回下一个节点名）：

```python
def merge_chunks(old: list[Chunk], new: list[Chunk]) -> list[Chunk]:
    """并行分支写入 candidates 时的 reducer：按 chunk_id 去重，保留最高分。"""
    by_id = {c["chunk_id"]: c for c in old}
    for c in new:
        prev = by_id.get(c["chunk_id"])
        if prev is None or c["score"] > prev["score"]:
            by_id[c["chunk_id"]] = c
    return list(by_id.values())

def merge_usage(old: dict, new: dict) -> dict:
    return {k: old.get(k, 0) + new.get(k, 0) for k in set(old) | set(new)}

def pick_route(s: RAGState) -> str:
    return s["route"]                    # direct | single | multi | agent

def needs_grade(s: RAGState) -> str:
    return "grade" if s.get("use_grade") else "generate"

def after_fuse(s: RAGState) -> str:
    if s.get("use_rerank"):
        return "rerank"
    return "grade" if s.get("use_grade") else "generate"

def needs_critique(s: RAGState) -> str:
    return "critique" if s.get("use_critique") and s.get("contexts") else "finalize"

def after_grade(s: RAGState) -> str:
    if s["verdict"] == "correct":
        return "generate"
    if s["round"] >= s["max_rounds"]:
        return "generate"                # 带弃答提示
    return "expand" if s["verdict"] == "ambiguous" else "rewrite"

def after_critique(s: RAGState) -> str:
    if s["supported"] == "full":
        return "finalize"
    if s["round"] >= s["max_rounds"]:
        return "finalize"                # 输出“证据不足”
    return "rewrite"

def after_agent(s: RAGState) -> str:
    if s["answer"] or s["round"] >= s["max_rounds"]:
        return "finalize"
    return "tool_search"
```

```python
g = StateGraph(RAGState)
for name, fn in [("prepare", prepare), ("route", route), ("expand", expand),
                 ("retrieve", retrieve), ("fuse", fuse), ("rerank", rerank),
                 ("grade", grade), ("generate", generate), ("critique", critique),
                 ("rewrite", rewrite), ("agent", agent),
                 ("tool_search", tool_search), ("finalize", finalize)]:
    g.add_node(name, fn)

g.add_edge(START, "prepare")
g.add_edge("prepare", "route")
g.add_conditional_edges("route", pick_route,
                        {"direct": "generate", "single": "retrieve",
                         "multi": "expand", "agent": "agent"})
g.add_edge("expand", "retrieve")
g.add_edge("retrieve", "fuse")       # 单路检索时 fuse 直接透传
g.add_conditional_edges("fuse", after_fuse,
                        {"rerank": "rerank", "grade": "grade", "generate": "generate"})
g.add_conditional_edges("rerank", needs_grade,
                        {"grade": "grade", "generate": "generate"})
g.add_conditional_edges("grade", after_grade,
                        {"generate": "generate", "expand": "expand", "rewrite": "rewrite"})
g.add_conditional_edges("generate", needs_critique,
                        {"critique": "critique", "finalize": "finalize"})
g.add_conditional_edges("critique", after_critique,
                        {"finalize": "finalize", "rewrite": "rewrite"})
g.add_edge("rewrite", "retrieve")
g.add_edge("tool_search", "agent")
g.add_conditional_edges("agent", after_agent,
                        {"tool_search": "tool_search", "finalize": "finalize"})
g.add_edge("finalize", END)
```

RAG-Fusion 的多路检索用 `Send` 做 map-reduce（LangGraph 官方文档明确支持 `Send` 对象做 fan-out），每个 `(query, index)` 组合发一个 `retrieve` 任务，`fuse` 等所有分支结束后再跑。

### 5.5 每一级如何“接线”

不必为每一级改图，只改 `strategy` 对应的配置：

```python
MILESTONES = {
  "L1_bm25":    dict(route="single", index="bm25",      expand="none",        fuse=False, rerank=False, grade=False, critique=False),
  "L1_dense":   dict(route="single", index="dense",     expand="none",        fuse=False, rerank=False, grade=False, critique=False),
  "L2_hybrid":  dict(route="single", index="hybrid",    expand="none",        fuse=True,  rerank=True,  grade=False, critique=False),
  "L3_hyde":    dict(route="single", index="dense",     expand="hyde",        fuse=False, rerank=True,  grade=False, critique=False),
  "L3_fusion":  dict(route="multi",  index="hybrid",    expand="multi_query", fuse=True,  rerank=True,  grade=False),
  "L4_ctx":     dict(route="single", index="contextual",expand="none",        fuse=True,  rerank=True,  grade=False),
  "L5_raptor":  dict(route="single", index="raptor",    expand="none",        fuse=False, rerank=True,  grade=False),
  "L6_crag":    dict(route="single", index="hybrid",    expand="none",        fuse=True,  rerank=True,  grade=True,  critique=False),
  "L6_selfrag": dict(route="single", index="hybrid",    expand="none",        fuse=True,  rerank=True,  grade=True,  critique=True),
  "L7_adapt":   dict(route="auto",   index="auto",      expand="auto",        fuse=True,  rerank=True,  grade=True,  critique=True),
  "L8_agent":   dict(route="agent",  index="hybrid",    expand="none",        fuse=True,  rerank=True,  grade=False, critique=False),
}
```

`route="auto"` 时，`route` 节点先做复杂度判断，再把 `state["route"]` 改成 `direct/single/multi`；`index="auto"` 时，简单题走 dense，实体题走 hybrid，全局题走 raptor。**这样 L7 不是另一条管道，而是“会改自己 state 的路由器”**——这正是 LangGraph 相对于 `if/else` 脚本的优势。

> 实现细节：`MILESTONES` 里的 `"auto"` 只是**配置输入**，不是 state 的合法值。`prepare` / `route` 节点必须先把它解析成 `RAGState` 允许的 `Literal` 值，否则 `pick_route` 会返回 `"auto"` 而找不到目标节点。同理，`fuse=False` 时 `fuse` 节点要做单列表透传，`grade=False` / `critique=False` 时由 `needs_grade` / `needs_critique` 直接跳过。

### 5.6 建索引图（离线）

```
chunks.jsonl
  → contextualize（LLM，仅 L4 需要）
  → embed（bge-m3，免费）
  → build_bm25（jieba + bm25s）
  → build_raptor（仅 L5 需要：聚类 + 摘要 + 再嵌入）
  → persist（parquet + faiss + bm25 索引 + raptor 树）
```

这张图只跑一次，产物落盘；查询图只读。把“索引”与“查询”分成两张图，可以避免“为了跑 L4 又把 L1 的索引重算一遍”这种低级浪费。

### 5.7 持久化与缓存

- **checkpointer**：评测时每个 `qid` 一个 `thread_id`，用 `SqliteSaver`（或 `InMemorySaver` 提速）。好处是轨迹可回放、失败可续跑；LangGraph 官方文档明确 `thread_id` 是 checkpoint 的主键。
- **LLM 缓存**：在节点边界按 `hash(model, prompt, temperature, max_tokens)` 缓存到磁盘。重跑同一级时直接命中，成本接近 0。这是把 $10 预算变成“可以反复调试”的关键。
- **检索缓存**：按 `(index_name, query, k)` 缓存。HyDE 与 RAG-Fusion 都会重复调用同一个 dense 检索，缓存能省掉重复的 embedding 请求（虽然免费，但省时间）。
- **不要在 state 里放大对象**：`contexts` 只放 `chunk_id + score + text`，向量放外部存储；否则 checkpoint 会变得很大。

---

## 6. 成本总表

### 6.1 单级成本（150 题，含索引）

| 级别 / 变体 | 索引 LLM | 单题 | 150 题查询 | 150 题 judge | 小计 |
|---|---:|---:|---:|---:|---:|
| L1a BM25 | $0 | $0.000534 | $0.080 | $0.081 | **$0.161** |
| L1b Dense | $0 | $0.000534 | $0.080 | $0.081 | **$0.161** |
| L2 Hybrid+RRF+Rerank | $0 | $0.000606 | $0.091 | $0.081 | **$0.172** |
| L3a HyDE | $0 | $0.000710 | $0.107 | $0.081 | **$0.188** |
| L3b RAG-Fusion | $0 | $0.000686 | $0.103 | $0.081 | **$0.184** |
| L4 Contextual | $0.15–0.37 | $0.000606 | $0.091 | $0.081 | **$0.322–0.542** |
| L5 RAPTOR | $0.08 | $0.000536 | $0.080 | $0.081 | **$0.241** |
| L6a Self-RAG | $0 | $0.001520 | $0.228 | $0.081 | **$0.309** |
| L6b CRAG | $0 | $0.000579 | $0.087 | $0.081 | **$0.168** |
| L7 Adaptive | $0 | $0.000712 | $0.107 | $0.081 | **$0.188** |
| L8 Agentic | $0 | $0.002000 | $0.300 | $0.081 | **$0.381** |

> judge 的口径：一次调用同时给问题、gold、答案、6 个上下文，约 3,000 in / 80 out ≈ $0.00054/题；150 题 ≈ $0.081。若只抽样 30 题，按 1/5 计。

### 6.2 推荐组合的总成本

**主推组合（8 级、11 个变体全跑）**：

| 项 | 金额 |
|---|---:|
| 查询（11 个变体 × 150 题） | $1.354 |
| 索引（L4 $0.25 + L5 $0.08） | $0.33 |
| judge（11 × $0.081） | $0.891 |
| **小计** | **$2.58** |
| 30% 重跑/重试余量 | $0.77 |
| **总计** | **≈ $3.4** |

**只跑 8 个“一级各一个变体”**：查询 $1.274 + 索引 $0.33 + judge $0.648 ≈ **$2.25**；加 30% 余量 ≈ **$2.9**。

**当前 30 题试点集**：查询与 judge 都按 1/5 计，约 **$0.45 + $0.33 索引 ≈ $0.78**；加余量 ≈ **$1.0**。这也说明：**应该先在 30 题上把 8 级全部跑通，再补题到 150**，而不是一上来就烧 150 题。

### 6.3 预算安全性

- 即使 L4 按最贵的“每 chunk 一次”($0.37)、L8 按 6 步上限($0.45)、judge 全量，总成本仍 **< $4**。
- 若把 FLARE（+$0.25）、Speculative（+$0.24）、Lost-in-the-Middle 三种顺序（+$0.05）都加上，仍 **< $5**。
- **$10 的真实约束不是钱，而是“不要重复索引”**：L4 的 869 次上下文生成如果每次调试都重跑，很容易烧掉 $3–5。必须做磁盘缓存。
- SiliconFlow 的 ¥10 基本用不到：bge-m3 与 bge-reranker 都是免费档；只有做 embedding/rerank 模型对比消融时才需要 Pro 档（¥0.07/M）。

### 6.4 评测成本与协议

- **确定性指标（$0）**：EM/F1（中文按字/词）、`evidence_recall`（gold doc 的 chunk 是否进入最终上下文）、`citation_precision`、`abstain_accuracy`。
- **LLM judge（$0.081/150 题/级）**：correctness + faithfulness；同一模型、`temperature=0`、同一 prompt，A/B 顺序对调各跑一次。
- **RAGAS**：只取 faithfulness 与 context precision 作为补充；不要在 8 级 × 150 题上全量跑所有指标，成本会翻倍。
- **统一纪律**：同一份 `chunks.jsonl`、同一个 `chunk_id`、同一个 `MAX_CONTEXT_TOKENS=4000`、同一个 judge、同一批题；否则跨级不可比。

---

## 7. 共享缓存矩阵

| 产物 | 谁在用 | 是否可复用 | 备注 |
|---|---|---|---|
| `chunks.jsonl`（869） | 全部 | ✅ | 冻结，禁止重切 |
| bge-m3 基础向量（869 × 1024） | L1/L2/L3/L6/L7/L8 | ✅ | 免费，但重算费时间 |
| BM25 索引（jieba） | L1a/L2/L3b/L6/L7/L8 | ✅ | 与 dense 同一批 chunk |
| Contextual 前缀（869 条说明） | L4 | ❌ 仅 L4 | 生成一次，落盘 |
| Contextual 向量 + Contextual BM25 | L4 | ❌ 仅 L4 | 与基础索引并存，不覆盖 |
| RAPTOR 树（约 100 个摘要节点）+ 节点向量 | L5 | ❌ 仅 L5 | 叶子复用基础向量，只有摘要节点新嵌入 |
| reranker | L2–L8 | ✅ | 无状态，免费 |
| query 向量 | 全部 | ✅ | 按 query 文本缓存 |
| LLM 答案 | 全部 | ✅ | 按 `(strategy, qid, prompt_hash)` 缓存 |
| judge 结果 | 评测 | ✅ | 按 `(qid, answer_hash, rubric_version)` 缓存 |
| gold 展开的 `gold_chunk_ids` | 评测 | ✅ | 由 `doc_id` + 切分规则自动生成，不要手写 |

**结论**：L1/L2/L3/L6/L7/L8 共享同一套 chunk + 向量 + BM25，边际索引成本为 $0；只有 L4 需要第二套 contextual 索引，L5 需要一棵摘要树。这就是总成本能压在 $3 左右的原因。

---

## 8. 风险与实现陷阱

1. **天花板效应（最大的方法论风险）**：869 个 chunk，top-5 只覆盖 0.58% 的语料，检索很容易命中；很多级别的差异会很小，看起来“什么都没用”。对策：按题型分层报告、使用 `evidence_recall` 而不是 recall@k、把无答案题（当前的 `none`，补题后的 `no_answer`）与 `global` 单独列出、必要时构造“难负例”子集。不要用 30 题的平均分下结论。
2. **评测集还没到 150 题**：当前只有 30 题，且类型名是 `multi`/`none`。先改代码适配，再补题；补题时保持 `qid` 稳定，否则历史结果无法对齐。
3. **Self-RAG 不是 Self-RAG**：没有 reflection token 的训练，只有 prompt 模拟。报告里必须写“Self-RAG-style prompting”，并说明原论文的关键贡献（critic 数据 + 端到端训练）没有被复现。
4. **CRAG 没有 web**：离线评测不能用网页搜索，否则不可复现。用“改写 + 换检索器 + 放宽 k”作为 Incorrect 分支的替身，并明确标注。
5. **Adaptive-RAG 没有分类器**：零样本路由会有错分；把“路由准确率”本身作为一个指标报出来，而不是假装它总是对的。
6. **FLARE 的 logprobs**：DeepSeek 支持 `logprobs`，但不同 SDK 的字段可能不同；先写一个 10 行脚本验证 `choices[0].logprobs.content[*].logprob` 能拿到，再接进图。若拿不到，退化为 `FLARE_instruct`（让模型输出 `[Search(...)]`），并在报告里说明。
7. **Contextual Retrieval 的成本失控**：按 chunk 逐条调用会把输入放大到 2.26M token；一定按文档批量，并把结果按 `chunk_id` 落盘。调试时先跑 10 篇，确认 prompt 稳定再全量。
8. **RAPTOR 在 200 篇上的树太浅**：869 个叶子 → 约 87 + 9 + 1 个节点，只有 3 层；全局题可能仍然“摘要太薄”。可以增大每簇节点数（12–15）或只建两层，并在报告里说明这是规模限制，不是方法失败。
9. **中文 BM25 的分词**：必须用 jieba，且索引与查询用**同一个** tokenizer 实例；停用词表要包含中文虚词。否则 BM25 的分数会不可比。
10. **引用与弃答的统一**：所有级别必须输出 `chunk_id` 引用；弃答阈值 τ 要在一份独立 dev 集上标定，然后**套用到所有级别**，否则 `abstain_accuracy` 不可比。
11. **不要把“做不到”记成 0 分**：例如 L1 在 global 题上不是“答错”，而是“没有这个能力”；报告里用 `N/A`，并解释原因。
12. **成本记账**：每个节点写 `usage`，每次运行落一份 JSON；报告里的成本必须是**实测 token**，不是本文估算。本文的 $2.3–3.0 只用于立项判断。

---

## 9. 一页纸的“历史时间线”（用于报告开头）

| 年份 | 工作 | 在本项目中的位置 |
|---|---|---|
| 2009 | Cormack et al., RRF | L2 的融合算法 |
| 2020 | Lewis et al., RAG | L1 的思想源头 |
| 2020 | Karpukhin et al., DPR | L1 的 dense 检索器 |
| 2022 | Gao et al., HyDE | L3 变体 A |
| 2022 | Yao et al., ReAct | L8 |
| 2023 | Jiang et al., FLARE | L6/L8 的可选开关 |
| 2023 | Asai et al., Self-RAG | L6 变体 A |
| 2023 | Liu et al., Lost in the Middle | L2 的诊断 |
| 2023 | Gao et al., RAG Survey（Naive/Advanced/Modular） | 全项目的叙事框架 |
| 2024 | Yan et al., CRAG | L6 变体 B |
| 2024 | Sarthi et al., RAPTOR | L5 |
| 2024 | Jeong et al., Adaptive-RAG | L7 |
| 2024 | Anthropic, Contextual Retrieval | L4 |
| 2024 | Rackauckas, RAG-Fusion | L3 变体 B |
| 2024 | Edge et al., GraphRAG | 排除，但在报告中讲清楚 |
| 2024 | Günther et al., Late Chunking | 排除，但讲清楚与 L4 的关系 |
| 2024 | Wang et al., Speculative RAG | 排除，低优先级 |
| 2023/2024 | Es et al., RAGAS | 评测工具 |

---

## 10. 参考链接

**论文 / 博客（按首次出现顺序）**

1. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020 — https://arxiv.org/abs/2005.11401
2. Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020 — https://arxiv.org/abs/2004.04906 ；https://aclanthology.org/2020.emnlp-main.550/
3. Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009 — https://doi.org/10.1145/1571941.1572114 ；https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
4. Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009 — https://doi.org/10.1561/1500000019
5. Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE), 2022 — https://arxiv.org/abs/2212.10496 ；https://aclanthology.org/2023.acl-long.99/
6. Rackauckas, *RAG-Fusion: a New Take on Retrieval-Augmented Generation*, 2024 — https://arxiv.org/abs/2402.03367
7. Anthropic, *Introducing Contextual Retrieval*, 2024-09-19 — https://www.anthropic.com/research/contextual-retrieval ；实验附录：https://www-cdn.anthropic.com/5722e7658c9302d8b97a3238de1bb8e6afdf04b9.pdf
8. Jiang et al., *Active Retrieval Augmented Generation* (FLARE), EMNLP 2023 — https://arxiv.org/abs/2305.06983 ；https://aclanthology.org/2023.emnlp-main.495/
9. Asai et al., *Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection*, ICLR 2024 — https://arxiv.org/abs/2310.11511 ；https://selfrag.github.io/
10. Yan et al., *Corrective Retrieval Augmented Generation* (CRAG), 2024 — https://arxiv.org/abs/2401.15884
11. Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*, ICLR 2024 — https://arxiv.org/abs/2401.18059
12. Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023 — https://arxiv.org/abs/2210.03629
13. Singh et al., *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG*, 2025 — https://arxiv.org/abs/2501.09136
14. Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*, NAACL 2024 — https://arxiv.org/abs/2403.14403 ；https://aclanthology.org/2024.naacl-long.389/
15. Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, 2024 — https://arxiv.org/abs/2404.16130
16. Microsoft GraphRAG, *Indexing Methods* — https://microsoft.github.io/graphrag/index/methods/
17. Microsoft Research, *LazyGraphRAG: Setting a new standard for quality and cost*, 2024 — https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/
18. Günther et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*, 2024 — https://arxiv.org/abs/2409.04701 ；https://jina.ai/news/late-chunking-in-long-context-embedding-models/
19. Wang et al., *Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting*, 2024 — https://arxiv.org/abs/2407.08223
20. Es et al., *RAGAs: Automated Evaluation of Retrieval Augmented Generation*, EACL 2024 Demo — https://arxiv.org/abs/2309.15217 ；https://aclanthology.org/2024.eacl-demo.16/
21. Ragas 文档, *Context Recall* — https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/
22. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024 — https://arxiv.org/abs/2307.03172 ；https://aclanthology.org/2024.tacl-1.9/
23. Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey*, 2023 — https://arxiv.org/abs/2312.10997
24. Fazlija et al., *Evaluating RAG-Fusion with RAGElo: an Automated Elo-based Framework*, 2024 — https://arxiv.org/abs/2406.14783

**模型与 API 规格**

25. BAAI/bge-m3 模型卡（1024 维 / 8192 token / dense+sparse+multi-vector） — https://huggingface.co/BAAI/bge-m3
26. BGE M3-Embedding 论文 — https://arxiv.org/abs/2402.03216
27. SiliconFlow, bge-reranker-v2-m3 模型页（免费档） — https://www.siliconflow.cn/models/BAAI%2Fbge-reranker-v2-m3
28. DeepSeek API, Chat Completions（`logprobs` / `top_logprobs`） — https://api-docs.deepseek.com/api/create-chat-completion/

**LangGraph 文档**

29. LangGraph Graph API（StateGraph、conditional edges、Send） — https://docs.langchain.com/oss/python/langgraph/graph-api
30. LangGraph Subgraphs（共享 state schema） — https://docs.langchain.com/oss/python/langgraph/use-subgraphs
31. LangGraph Persistence（checkpointer / store） — https://docs.langchain.com/oss/python/langgraph/persistence
32. LangGraph Checkpointers（`thread_id`） — https://docs.langchain.com/oss/python/langgraph/checkpointers

**本地数据（非 URL，供复核）**

33. `C:\Users\Ashit\Documents\GitHub\rag-ladder\data\manifest.json`
34. `C:\Users\Ashit\Documents\GitHub\rag-ladder\data\manifest-chunks.json`
35. `C:\Users\Ashit\Documents\GitHub\rag-ladder\data\chunks.jsonl`（字段样例）
36. `C:\Users\Ashit\Documents\GitHub\rag-ladder\eval\queries.jsonl`（30 题现状）

---

## 附录 A：与 `rag-ladder/search/milestones.md` 旧方案的关系

旧方案是 **2,000 篇 / M1–M6 / GraphRAG 为主角**；本方案是 **200 篇 / L1–L8 / GraphRAG 排除**。差异不是“缩水”，而是换了教学重点：

| 维度 | 旧方案（rag-ladder） | 新方案（baby-rag-history） |
|---|---|---|
| 语料 | 2,000 篇 / 10k chunk | 200 篇 / 869 chunk |
| 主线 | 六种“范式”同台 | 八个“控制点”递进 |
| GraphRAG | 核心（M4） | 排除，作为反面教材 |
| 查询侧 | 基本没展开 | HyDE + RAG-Fusion（L3） |
| 索引侧 | Contextual 并入 M2 | Contextual 独立成 L4 |
| 控制流 | CRAG + Agentic | Self-RAG + CRAG + Adaptive + Agentic |
| 成本 | 一轮 $13.8（2,000 篇） | 一轮 $2.3–3.4（200 篇） |

旧方案里仍然有效的纪律应当继承：**同一份 chunk、同一批题、同一个 judge、`MAX_CONTEXT_TOKENS=4000`、`evidence_recall` 替代 recall@k、弃答 τ 统一、成本三项必报**。

## 附录 B：建议的目录结构

```
baby-rag-history/
├─ data/                         # git-ignored，指向 rag-ladder/data 或复制
├─ research/
│   └─ milestones.md             # 本文
├─ src/baby_rag/
│   ├─ state.py                  # RAGState + reducers
│   ├─ build_graph.py            # 离线索引图
│   ├─ query_graph.py            # 在线查询图
│   ├─ nodes/                    # retrieve/fuse/rerank/generate/critique/agent
│   ├─ indexes/                  # bm25.py dense.py contextual.py raptor.py
│   ├─ prompts/                  # 每个节点的中文 prompt，版本化
│   └─ eval/                     # metrics.py judge.py leaderboard.py
├─ configs/milestones.yaml       # §5.5 的配置表
├─ eval/queries.jsonl            # 先 30 题，后补到 150
└─ results/
    ├─ runs/                     # 每次运行的 trace + usage
    └─ leaderboard.md
```

**最后一个建议**：把“每个里程碑的预期失败”写进 `README`，而不是只写“预期提升”。历史项目的价值在于让读者看到**为什么下一步会出现**；如果每一级都只报“我们更好了”，它就退化成一张排行榜，而不是一段历史。
