# 评测协议

> 结果见 [`results.md`](results.md)；数字唯一来源是 `results/*/summary.json`。

## 1. 指标

| 指标 | 定义 | 用途 | 注意 |
|---|---|---|---|
| **可答 acc** | 可回答问题中，judge 判为正确的比例 | 端到端效果 | 所有 rung 同一分母（原题 27 / 难题 25） |
| **弃答 acc** | 无答案题中，输出「无法回答」的比例 | 拒答能力 | 所有 rung 都报；只有 r7/r8/adaptive 有专门弃答路径 |
| **evidence R@k** | gold evidence 是否进入最终 top-k 上下文 | 检索召回 | k = 最终上下文条数 5，不用 @10 |
| **doc all@5** | **所有** gold 文档是否都进 top-5 | 多跳严格召回 | 单跳题与 doc R@5 相同 |
| **doc@1** | 第一个命中是否来自 gold 文档 | 粗粒度诊断 | 容易饱和，不作主指标 |
| **MRR@10** | gold evidence 第一次出现的排名倒数 | 排序质量 | 对首轮排序敏感 |
| **nDCG@10** | 二值相关下的排序质量 | 排序质量 | 与 MRR 高度相关 |
| **cold $/q** | token × 单价，按冷启动算 | 成本 | 不受缓存影响 |
| **faithfulness** | 答案中的事实是否被引用资料支持 | 幻觉/引用 | 抽样 30 条，单 judge，区分度低 |

`N/A` 表示该档不具备该能力，或该题没有对应 gold evidence，**不是 0 分**。

## 2. 协议 v2

- **同语料、同问题、同预算**：所有 rung 共用 chunks、问题、top-5 / 4,000-token 上下文。
- **分母统一**：`可答 acc` 只算可回答问题；`弃答 acc` 只算 no-answer 题。两者不混算。
- **置信区间**：比例指标用 Wilson 95% CI（小样本比 bootstrap 稳）；bootstrap 结果保留在 `summary.json`。
- **judge 校准**：抽 30 条答案（15 个唯一问题），正常顺序与字段对调顺序各判一次，30/30 一致。
  这不是人工标注，也不排除 self-preference。
- **回顾性 split**：20 dev / 10 test；test 题在开发期已见过，只能作稳健性检查。
- **RAPTOR 公平性**：摘要节点按「覆盖 gold 文档」算命中，不要求摘要逐字包含 evidence。
- **外部锚点**：RAGBench `emanual` 25 题英文手册；只作方向对照，不与中文主榜混算。

## 3. 为什么主指标是 evidence R@1 / R@5，而不是 doc R@5

- 1,000 篇语料上 `doc R@5` 饱和到 100%，没有区分度。
- `evidence R@5` 是实际召回率：生成器只看 top-5，证据没进上下文就不可能被使用。
- `evidence R@1` 是最有区分度的排序指标：r5 与 r3 的差距在这里最明显。
- 多跳题要看 `doc all@5`：只看「任意一个 gold 文档」会高估。

## 4. 已知偏差

- **样本小**：原题 30、难题 25、test 10、中段事实 50、公开 25；1–2 道题的差异不构成结论。
- **标题泄漏**：原题 23/30 含 gold 标题；masked 变体去除标题并加 200 篇干扰文档。
- **masked 是近似**：contextual 前缀复用原前缀、替换标题后重新嵌入，没有重新跑 5,676 次生成。
  另外，chunk 文本去掉了标题，但 metadata 的 `title` 仍会由 `build_context` 拼回上下文，
  所以是**部分去偏**，不是完全去标题。
- **judge 偏差**：单模型、单次判定；A/B 只检查位置偏差，不检查事实判断错误。
- **弱模型对照的 judge 也是弱模型**：换 Qwen2.5-7B 时，判对错的也是它；只有 `evidence R@k` / MRR 是 judge 无关的。
- **API 错误**：弱模型跑出现 5 条 451/400/503，已从分母剔除；leaderboard 的 `err` 列会显示。
- **faithfulness 偏宽松**：30 条抽样里 29 条 `supported=1.0`，不能用来排名。
- **公开锚点小**：25 题英文手册，r5/r6 没有公共索引。
- **题库作者偏差**：25 道难题是项目作者手写的，可能偏向某些失败模式。
- **中段事实题偏向数字**：50 题由作者读 chunk 后反写，且选题时优先含数字的 chunk；
  它衡量“中段事实能否被检索到”，不代表真实用户的问题分布。

## 5. 复现

```bash
# 原题
uv run python -m scripts.run_eval --rungs all --variant original --workers 3

# 去标题
uv run python -m scripts.run_eval --rungs all --variant masked --workers 3

# 公平难题
uv run python -m scripts.run_eval --rungs all --queries eval/queries-hard.jsonl --workers 3

# 回顾性 test split
uv run python -m scripts.run_eval --rungs all --queries eval/queries-test.jsonl --workers 3

# 中段事实 50 题（手写）
uv run python -m scripts.run_eval --rungs all --queries eval/queries-manual50.jsonl --workers 3

# 中段事实 50 题（弱模型）
uv run python -m scripts.run_eval --rungs all --queries eval/queries-manual50.jsonl --provider siliconflow --workers 3

# 弱模型对照（SiliconFlow 免费 Qwen2.5-7B）
uv run python -m scripts.run_eval --rungs all --provider siliconflow --workers 4

# 公开锚点（先构建 25 题公共索引）
uv run python -m scripts.build_public_benchmark
uv run python -m scripts.build_index --variant public --rebuild
uv run python -m scripts.run_eval --rungs r0_dense,r1_hybrid,r2_rerank,r4_multiquery,r7_crag,r8_agentic \
  --variant public --queries data/public/queries.jsonl --workers 3

# 从已有 per_query 重算指标 / 报告
uv run python -m scripts.recompute_metrics
```

## 6. 如果要更接近最佳实践

1. 冻结 100+ 题的 test 集，按题型分层，只跑一次。
2. 第二 judge（不同厂）+ 10–20% 人工抽检。
3. 公开集扩到 100+ 题，最好包含中文集（RGB / CRAG）。
4. 加 paired bootstrap / McNemar，报置信区间而不是单点。
5. faithfulness 用逐 claim 判断并人工校准，而不是整段一个分数。
