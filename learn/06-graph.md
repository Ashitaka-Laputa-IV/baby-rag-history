---
title: Graph 节点
date: 2026-11-23
input_shape: concept
subject: graph.py 的 Runner 与 build_graph
---

# Graph 节点

`graph.py` 是最大的一篇，但结构很规整：一个 `Runner` 类装所有节点，一个 `build_graph`
把它们连起来。读的时候按“节点组”读，不要从第一行顺读。

## Runner 与三个 helper

[graph.py:39](../src/babyrag/graph.py#L39)：

```python
class Runner:
    def __init__(self, services, llm): ...
    def _query(self, state):                 # 取最后一个 query variant
    def _record(self, node, result, note):   # LLMResult -> NodeRecord
    def _hybrid(self, query):                # dense + BM25
```

`_query`（[graph.py:45](../src/babyrag/graph.py#L45)）的行为很重要：

```python
variants = state.get("query_variants") or []
return variants[-1] if variants else state["question"]
```

HyDE 把“假答案”放在 `query_variants[-1]`，所以后续 `retrieve_hybrid` 自动用假答案检索。
Multi-query 则把原始问题放 `[0]`、改写放后面；`retrieve_multi` 遍历全部 variants，
不会只取最后一个。

## plan / dispatch / route

```mermaid
flowchart TD
  START --> plan --> dispatch
  dispatch -->|plan[cursor]| stage[任一新阶段节点]
  stage -->|cursor+1| dispatch
  dispatch -->|cursor >= len| finalize --> END
```

[plan:58](../src/babyrag/graph.py#L58) 只做两件事：

```python
if mode == "ladder":
    plan = list(RUNGS[rung])
else:
    rung, reason = adaptive_plan(question, qtype)
    plan = list(RUNGS[rung])
return {"plan": plan, "cursor": 0, ...}
```

[route:74](../src/babyrag/graph.py#L74) 是条件边的函数：

```python
return plan[cursor] if cursor < len(plan) else "finalize"
```

[dispatch:71](../src/babyrag/graph.py#L71) 返回 `{}`，只是同步点。

## 检索节点组

| 节点 | 做什么 | 写入 |
|---|---|---|
| `retrieve_dense` | dense k=10 | `hits` |
| `retrieve_hybrid` | dense + BM25 k=10 | `dense_hits`, `bm25_hits` |
| `retrieve_contextual` | contextual dense + contextual BM25 | `dense_hits`, `bm25_hits` |
| `retrieve_multi` | 对每个 variant 做 hybrid，再各自 RRF | `multi_hits` |
| `retrieve_raptor` | 叶子 + 摘要，RRF 融合 | `hits` |

对应代码：[dense:79](../src/babyrag/graph.py#L79) · [hybrid:84](../src/babyrag/graph.py#L84) ·
[contextual:90](../src/babyrag/graph.py#L90) · [multi:98](../src/babyrag/graph.py#L98) ·
[raptor:154](../src/babyrag/graph.py#L154)

`retrieve_raptor` 的注释值得读：

```python
# Collapsed tree: leaves and summary nodes are searched together, then fused.
# Summary-only retrieval loses every factual detail.
```

## 融合与精排

[fuse_rrf:107](../src/babyrag/graph.py#L107)：

```python
if multi: hits = fuse_rrf(*multi, top_n=20)
else:     hits = fuse_rrf(dense, bm25, top_n=20)
```

[rerank:118](../src/babyrag/graph.py#L118) 调 `rerank_hits`，`top_n=5`；如果 reranker
返回空（网络失败），回退到原 hits 的前 5 条。

## 查询变换

[hyde:126](../src/babyrag/graph.py#L126)：

```python
result = self.llm.invoke(HYDE.format(question=...), stage="hyde", ...)
return {"query_variants": [result.text.strip() or question], ...}
```

[rewrite_multi_query:134](../src/babyrag/graph.py#L134)：

```python
variants = [question] + parse_queries(result.text, 3)
```

`parse_queries`（[prompts.py:122](../src/babyrag/prompts.py#L122)）会去掉行首的 `-`、`•`、数字编号，
但不会校验语义；模型返回空行时可能少于 3 个变体。

## grade：CRAG 的核心

[graph.py:168](../src/babyrag/graph.py#L168)：

1. 取前 6 条 hits；没有就 `grade="incorrect"`。
2. 把每条截到 600 字，编号后一次性发给 LLM（**批量**，不是每条一次）。
3. `parse_relevant` 返回 `relevant` 序号和 `verdict`。
4. 如果 `verdict == "incorrect"` 且 `retries < 1`：改写问题 → hybrid 检索 → rerank，
   写入新 hits，`retries+1`。
5. 否则用 `relevant` 过滤 hits；如果解析失败，`parse_relevant` 默认全保留并判 `ambiguous`。

`grade` 的注释说明：这是“评分 + 一次重查”，不是无限循环；上限由 `retries < 1` 控制。

## agentic：四步 ReAct

[graph.py:198](../src/babyrag/graph.py#L198)：

```python
for hit in fuse_rrf(*self._hybrid(state["question"]), top_n=5):
    collected.setdefault(hit["chunk_id"], hit)
for step in range(4):
    result = llm.invoke(AGENTIC.format(question, history), stage="agentic", extra_key=str(step))
    ...
    query = ...  # "SEARCH: ..." 后面的内容
    hits = fuse_rrf(*self._hybrid(query), top_n=5)
    collected.setdefault(...)
```

三个细节：

- **先跑一次原始问题的检索**，否则早期改写可能把 gold chunk 挤掉（注释在
  [graph.py:202](../src/babyrag/graph.py#L202)）。
- `extra_key=str(step)` 让缓存键按步区分；同一问题重跑时每步仍能命中。
- 最多 4 步；模型输出 `FINAL:` 就提前结束。没有收集到任何 hit 时，回退一次 hybrid。

## generate / verify / finalize

[generate:230](../src/babyrag/graph.py#L230)：

```python
hits = (state.get("hits") or [])[:5]
context, kept = build_context(...)
if not kept: return {"answer": "无法回答", "abstain": True, ...}
result = llm.invoke(GENERATE..., system=SYSTEM_GROUNDED)
abstain = answer.startswith("无法回答") and len(answer) < 20
```

`verify`（[graph.py:249](../src/babyrag/graph.py#L249)）只做一件事：把 `abstain` 再确认一次，
目前没有引用合法性检查。`finalize`（[graph.py:255](../src/babyrag/graph.py#L255)）把
`node_records` 求和成 `cost`。

## build_graph

[graph.py:261](../src/babyrag/graph.py#L261)：

```python
builder = StateGraph(RAGState)
builder.add_node("plan", runner.plan)
builder.add_node("dispatch", runner.dispatch)
for name in STAGES: builder.add_node(name, getattr(runner, name))
builder.add_node("finalize", runner.finalize)

builder.add_edge(START, "plan")
builder.add_edge("plan", "dispatch")
builder.add_conditional_edges("dispatch", runner.route, mapping)
for name in STAGES: builder.add_edge(name, "dispatch")
builder.add_edge("finalize", END)
```

`mapping` 是 `{stage: stage}` 加 `{"finalize": "finalize"}`。条件边必须覆盖 `route` 可能返回的
所有名字，否则 LangGraph 会报错。

## run_one

[graph.py:281](../src/babyrag/graph.py#L281) 负责：

- 组装初始 state；
- `graph.invoke(..., config={"recursion_limit": 60})`；
- 把 state 整理成评测要的 dict，`cost["latency_ms"]` 用墙钟覆盖节点累计值。

## Check yourself

1. 为什么 `agentic` 要先跑一次原始问题的检索？
2. `grade` 在什么条件下会改写并重查？最多几次？
3. 为什么 `generate` 里又取了一次 `hits[:5]`，而不是信任上游？

### Answers

1. 防止模型第一步就改写查询、把 gold chunk 挤掉；baseline 先占住前 5 个位置，后续 `setdefault` 只补充不覆盖。
2. 当 `verdict == "incorrect"` 且 `retries < 1`；最多一次，之后无论结果如何都进入 `generate`。
3. 防御性约束：任何上游节点（agentic 收集、grade 过滤、raptor 融合）都可能返回超过 5 条；最终上下文必须统一 top-5，保证公平。
