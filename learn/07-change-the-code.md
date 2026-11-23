---
title: 动手改代码
date: 2026-11-23
input_shape: concept
subject: 跑起来、改行为、排查问题
---

# 动手改代码

前面六篇讲“它怎么工作”。这一篇讲“你想改的时候，从哪下手”。

## 先跑起来

```bash
uv sync
cp .env.example .env          # 填两个 key
uv run python -m pytest -q    # 7 个离线测试
uv run babyrag info           # 看 9 个 rung
uv run babyrag ask "开姆尼茨工业大学的前身创办于哪一年？" --rung r2_rerank
```

`data/` 默认不在仓库里。想重建，看 [scripts/](../scripts) 下的脚本顺序；
如果只是读代码，不需要数据也能看 `pytest` 和 `babyrag info`。

## 文件地图

| 想找什么 | 去哪 |
|---|---|
| 一次请求怎么走 | [graph.py](../src/babyrag/graph.py) |
| 流程里传什么 | [types.py](../src/babyrag/types.py) |
| 路径、价格、开关 | [config.py](../src/babyrag/config.py) |
| 检索器 | [retrieval.py](../src/babyrag/retrieval.py) |
| 九个 rung 的工序单 | [rungs.py](../src/babyrag/rungs.py) |
| 提示词与解析 | [prompts.py](../src/babyrag/prompts.py) |
| 模型调用与缓存 | [llm.py](../src/babyrag/llm.py)、[cache.py](../src/babyrag/cache.py) |
| 指标与报告 | [eval.py](../src/babyrag/eval.py) |
| 命令行 | [cli.py](../src/babyrag/cli.py) |
| 数据/索引/评测脚本 | [scripts/](../scripts) |

## 改法一：让上下文看更多或更少

现在写死了两处：

```python
# graph.py:52  retrieve_hybrid 各取 10
dense = self.services.dense.search(query, k=10)
bm25 = self.services.bm25.search(query, k=10)

# graph.py:232  generate 只留 5
hits = (state.get("hits") or [])[:5]
```

另外 `rerank` 里还有 `top_n=5`（[graph.py:118](../src/babyrag/graph.py#L118)）。

想改成 top-3：

1. 在 [config.py:71](../src/babyrag/config.py#L71) 加 `FINAL_TOP_K = 3`。
2. 把 `generate` 的 `[:5]` 改成 `[:FINAL_TOP_K]`。
3. 把 `rerank` 的 `top_n=5` 也改成 `FINAL_TOP_K`，否则上游仍给 5 条。
4. 重跑评测，报告里的说明也要改。

**只改一处会怎样**：只改 `generate`，上游仍返回 5 条，只是最后截断；只改 `rerank`，
`generate` 仍取 5 条。三处要一起改。

## 改法二：加一个 rung

最简单的 rung 只改 [rungs.py:5](../src/babyrag/rungs.py#L5)：

```python
"r9_hybrid_no_rerank": ["retrieve_hybrid", "fuse_rrf", "generate"],
```

然后在 `LABELS` / `PAPERS` 加名字和论文。`ALL_RUNGS` 会自动包含它。

如果要用**新节点**，还要：

1. 在 [graph.py:32](../src/babyrag/graph.py#L32) 的 `STAGES` 加名字；
2. 在 `Runner` 里写同名方法；
3. `build_graph` 会自动注册，不用改拓扑。

## 改法三：加一个指标

以“答案引用的卡片里，至少一个属于标准文章”为例：

1. 在 [eval.py:116](../src/babyrag/eval.py#L116) 的 `_score` 里，用 `result["citations"]`
   和 `hits` 建映射，判断是否和 `gold_docs` 相交。
2. 在 [eval.py:177](../src/babyrag/eval.py#L177) 的 `summarize` 里用 `_mean` 汇总。
3. 在 [eval.py:275](../src/babyrag/eval.py#L275) 的 `write_reports` 加一列。
4. 在 `docs/evaluation.md` 补定义。

## 改法四：修 masked 的标题泄漏

现在 masked 只替换了卡片**正文**里的标题，但卡片还带着原来的 `title`，
拼上下文时会作为前缀出现。

想真正去掉：

1. 在 [build_masked_corpus.py](../scripts/build_masked_corpus.py) 里，把 masked 卡片的
   `title` 也设为「（标题已隐藏）」。现在只改了正文：

```python
masked = dict(chunk)
masked["text"] = mask_text(chunk["text"], chunk["title"])
# 还缺一行：
masked["title"] = "（标题已隐藏）"
```
2. 重建 `chunks.masked.jsonl`。
3. 重建 `chunks_masked` 集合（embedding 免费）。
4. 重跑 masked 评测，比较前后数字。

这是一个真实的、值得做的修正；做完后报告里要明确写“完全去标题”，而不是“部分去偏”。

## 出问题时先看哪

| 现象 | 先看 |
|---|---|
| 启动报 `OPENCODE_GO_API_KEY is not set` | `.env` 是否在仓库根目录 |
| 请求 400，提示缺 session | `config.py` 的 `session_id` 是否为空 |
| 检索结果全是空的 | `data/` 下有没有对应变体的 chunks/index |
| 评测大批 429 | 降低 `--workers`；`llm.py` 已有退避 |
| 报告里 `N/A` | 该题没有 gold evidence，或该 rung 不具备该能力 |
| 结果和上次不一样 | 先看 `summary.json` 的 `cache_hits`，可能命中了旧缓存 |

## 如果你只记一件事

> 改行为先找 `config`，改流程先找 `rungs`，改某一步先找 `graph`，
> 改指标先找 `eval`；改完先跑 `pytest`，再跑 3 道题冒烟。

## 想深入看代码

- 工序单：[rungs.py:5](../src/babyrag/rungs.py#L5)
- 节点实现：[graph.py:39](../src/babyrag/graph.py#L39)
- 指标：[eval.py:116](../src/babyrag/eval.py#L116)
- 数据脚本：[scripts/](../scripts)
- 测试：[tests/test_core.py](../tests/test_core.py)

## 检查一下

**问：** 想把“最多看 5 张卡片”改成 3 张，最少要改哪几处？

**答：** `config` 加常量、`rerank` 的 `top_n`、`generate` 的截断；只改一处会导致上下游不一致。
