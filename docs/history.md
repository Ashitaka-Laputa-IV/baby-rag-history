# RAG 里程碑简史（本项目实现的九档）

> 一句话：每一档只回答三个问题——**多了什么节点、贵了多少、还有什么做不到。**

## r0 · Naive RAG（2020）

- **论文**：Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- **机制**：query → bge-m3 向量 → top-5 → 拼进 prompt → 生成。
- **意义**：把「检索」和「生成」端到端接起来，证明外部知识可以补参数化记忆。
- **在本项目**：所有提升的分母。它已经能答对大部分单跳题，这正是天花板效应的来源。

## r1 · Hybrid + RRF（2009）

- **论文**：Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009，[DOI](https://doi.org/10.1145/1571941.1572114)；Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009。
- **机制**：BM25（jieba 分词）与 dense 两路召回，按 `Σ 1/(60+rank)` 融合。
- **意义**：稠密检索擅长语义，稀疏检索擅长专名/编号；两者互补。
- **在本项目**：不增加任何 LLM 调用，成本和 r0 相同。

## r2 · Rerank（2019）

- **论文**：Nogueira & Cho, *Passage Re-ranking with BERT*, [arXiv:1901.04085](https://arxiv.org/abs/1901.04085)；本项目用 bge-reranker-v2-m3。
- **机制**：先召回 20，再用 cross-encoder 精排取 5。
- **意义**：把「召回」和「排序」拆开；rerank 往往是单次增益最大的工程件。
- **在本项目**：SiliconFlow 免费，成本仍与 r0 相同，主要代价是延迟。

## r3 · HyDE（2022）

- **论文**：Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- **机制**：先让 LLM 写一段「假答案」，embed 它而不是原问题。
- **意义**：当问题与文档措辞差异大时，假答案更接近文档分布。
- **在本项目**：每题多一次生成调用；假答案可能编造年份/人名，这是它的真实风险。

## r4 · Multi-Query / RAG-Fusion（2024）

- **论文**：Rackauckas, *RAG-Fusion: a New Take on Retrieval-Augmented Generation*, [arXiv:2402.03367](https://arxiv.org/abs/2402.03367)
- **机制**：1 问改写为 3 问，分别检索后 RRF 融合。
- **意义**：用查询多样性对抗单词命中的不稳定。
- **在本项目**：每题多一次改写调用 + 3 路检索。

## r5 · Contextual Retrieval（2024）

- **来源**：Anthropic, *Introducing Contextual Retrieval*, 2024-09-19，[博客](https://www.anthropic.com/news/contextual-retrieval)
- **机制**：索引期为每个 chunk 生成 50–100 token 的文档内定位说明，拼在原文前再 embed / 建 BM25。
- **意义**：chunk 被切碎后丢失的指代与语境，可以在索引期补回来。
- **在本项目**：一次性索引成本，查询期不再多花钱；这是「索引侧」与「查询侧」的分水岭。

## r6 · RAPTOR（2024）

- **论文**：Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*, [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)
- **机制**：叶子 chunk → 聚类 → LLM 摘要成父节点 → 递归；查询时跨层检索（collapsed tree）。
- **意义**：全局综述题需要「摘要层」，而不是更多原始片段。
- **在本项目**：用约 1/14 的 GraphRAG 成本覆盖同类全局题；验证「全局问题不一定需要图」。

## r7 · CRAG / Self-RAG（2023–2024）

- **论文**：Yan et al., *Corrective Retrieval Augmented Generation*, [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)；Asai et al., *Self-RAG*, [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- **机制**：检索后先评估相关性（correct / ambiguous / incorrect），不足则改写重查，仍不足则弃答。
- **意义**：RAG 不只需要「查得准」，还需要「知道自己不知道」。
- **在本项目**：成本约是 r0 的 2.4 倍；必须同时报「误弃率」。

## r8 · Agentic ReAct（2022）

- **论文**：Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)；综述 [arXiv:2501.09136](https://arxiv.org/abs/2501.09136)
- **机制**：`Thought → Action(search) → Observation` 循环，最多 4 步。
- **意义**：把检索从「一次性预处理」变成「模型可调用的工具」。
- **在本项目**：多跳题的主要收益来源；单跳题可能因过度检索变差。

## adaptive · Adaptive-RAG（2024）

- **论文**：Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*, [arXiv:2403.14403](https://arxiv.org/abs/2403.14403)
- **机制**：先判断问题复杂度，再选择不检索 / 单步 / 多步。
- **意义**：不是新能力，而是「何时用哪一档」。
- **在本项目**：规则优先（题型 + 长度），LLM 只做兜底；报 `routing_distribution` 与 `regret`。

## 不实现的档

- **GraphRAG**：成本上买得起（1,000 篇约 $3.5–6），但社区结构太薄、工程风险高。见 `graphrag-not-built.md`。
- **ColBERT / 多向量**：工程复杂度与收益不匹配。
- **微调 / 训练**：预算与时间不允许；DPR、Self-RAG 的 reflection token 都用同机制的现代替代。
