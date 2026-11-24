---
title: 代码漫游
date: 2026-11-24
input_shape: concept
subject: baby-rag-history 的逐函数代码阅读
---

# 代码漫游

[`../README.md`](../README.md) 是“先看懂”，这套是“真的读代码”。

## 和通俗版的区别

| 通俗版 | 代码漫游 |
|---|---|
| 比喻、故事、结论 | 函数签名、逐行、调用关系 |
| 每篇一个概念 | 每篇一个文件或一组函数 |
| 代码只贴关键几行 | 贴出被讲函数的完整/半完整实现 |
| 适合第一次接触 | 适合准备改代码、做 review、排查 bug |

## 读法

1. **先看结构**：多数篇开头有模块的符号表或结构图，知道有哪些函数。
2. **按依赖顺序读**：地基 → 模型 → 检索 → 图 → 评测 → 脚本。
3. **每个函数问四件事**：拿什么、做什么、返回/改什么、谁调用。
4. **看到 `坑清单` / `易错点` 时停一下**：那是读这段代码时最容易踩的。
5. **最后做自测**：能答出来再进下一篇。

## 读代码时别被这些误导

下面这些名字会出现在源码里，但当前主流程**没有接线**：

| 名字 | 位置 | 状态 |
|---|---|---|
| `JsonCache` | `cache.py:69` | 没有调用者；主流程只用 `Cache` |
| `get_services` | `retrieval.py:163` | 没有调用者；评测用 `eval._env` 的线程本地 `Services` |
| `budget_left` | `llm.py:141` | 预留接口，图里没有调用 |
| `usd()` | `config.py:131` | 没有调用者；`llm.py` 内联了同样公式 |
| `PROMPT_VERSION` | `prompts.py:8` | 没有节点 import；`invoke` 默认 `"v1"` |
| `run_name` | `eval.py:71` | 参数未被 `evaluate` 使用；目录名由脚本直接传给 `write_reports` |

`expand_context` 已经注册成节点，但 `RUNGS` 里没有 rung 使用它；它是给“普通 chunk + contextual 前缀”预留的。

## 顺序

| # | 文档 | 读什么 | 前置 |
|---|---|---|---|
| 1 | [地基：config / types / cache](01-foundation.md) | 路径、价格、状态、SQLite | 无 |
| 2 | [模型调用与缓存](02-llm.md) | `LLMClient.invoke`、计费、重试 | 01 |
| 3 | [检索层](03-retrieval.md) | dense、BM25、RRF、rerank、上下文 | 01 |
| 4 | [Graph 上半场](04-graph-part1.md) | plan/dispatch、检索、查询变换 | 02, 03 |
| 5 | [Graph 下半场](05-graph-part2.md) | grade、agentic、generate、建图 | 04 |
| 6 | [评测](06-eval.md) | 打分、汇总、CI、报告 | 02, 04, 05 |
| 7 | [CLI 与脚本](07-scripts.md) | 入口与数据/索引管线 | 01–06 |

## 读之前

```bash
uv run python -m pytest -q     # 8 个纯逻辑测试
uv run babyrag info            # 看 RUNGS
```

测试是最短的“可执行规格”：`tests/test_core.py` 里覆盖了 RRF、解析器、rung 表、
variant 路径。

## 代码引用约定

- `[graph.py:58](../../src/babyrag/graph.py#L58)`：GitHub 会跳到第 58 行。
- “文档说” vs “代码是”：以代码为准；不一致会在那一节标出。
- 省略号 `...` 表示为了篇幅删掉的代码，不是源码里的字面量。

## 读完你应该能

- 画出 `ask` 的完整调用链，并说出每个阶段写入了哪些 state 字段。
- 解释 `LLMClient.invoke` 的缓存命中/未命中两条路径。
- 解释 RRF 为什么用排名而不是分数，以及 rerank 的下标映射。
- 解释 `grade` 的重查上限和 `agentic` 的 baseline 设计。
- 解释评测分母、Wilson 区间、综合分，以及它们各自不能证明什么。
- 知道新增 rung、改 top-k、加指标、修 masked 泄漏分别要动哪里。

## 自测

**问：** 这套漫游和 `learn/01–07` 应该先读哪个？

**答：** 先读通俗版建立地图，再读漫游。直接读漫游也能走，但会在术语和调用关系上更吃力。
