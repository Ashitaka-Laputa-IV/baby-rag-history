---
title: 代码漫游 04 · Graph 上半场：计划与检索
date: 2026-11-24
input_shape: concept
subject: graph.py 的 Runner 辅助函数、plan/dispatch 与检索节点
---

# 代码漫游 04 · Graph 上半场

这一篇读 [`graph.py`](../../src/babyrag/graph.py) 的前半段：
辅助函数、计划/派工、检索与查询变换节点。后半段的 `grade` / `agentic` / `generate`
在 [05](05-graph-part2.md)。

## 0. 结构

```
STAGES                   # 所有可注册的节点名
Runner
  _query / _record / _hybrid
  plan / dispatch / route
  retrieve_dense / retrieve_hybrid / retrieve_contextual / retrieve_multi
  fuse_rrf / rerank
  hyde / rewrite_multi_query / expand_context / retrieve_raptor
  （后半段）grade / agentic / generate / verify / finalize
build_graph / run_one
```

## 1. `STAGES`

[`graph.py:32`](../../src/babyrag/graph.py#L32)：

```python
STAGES = [
    "retrieve_dense", "retrieve_hybrid", "retrieve_contextual", "retrieve_multi",
    "retrieve_raptor", "hyde", "rewrite_multi_query", "fuse_rrf", "rerank",
    "expand_context", "grade", "agentic", "verify", "generate",
]
```

`build_graph` 会把它们全部注册成节点。`RUNGS` 里出现的名字必须在这里。
`expand_context` 目前没有 rung 使用，是给“普通 chunk + contextual 前缀”场景预留的。

## 2. Runner 的三个 helper

[`graph.py:39`](../../src/babyrag/graph.py#L39)：

```python
class Runner:
    def __init__(self, services: Services, llm: LLMClient):
        self.services = services
        self.llm = llm
```

### `_query`

[`graph.py:45`](../../src/babyrag/graph.py#L45)：

```python
variants = state.get("query_variants") or []
return variants[-1] if variants else state["question"]
```

- HyDE：`query_variants=[假答案]`，所以后续检索用假答案。
- Multi-query：`variants=[原问题, 改写1, 改写2, ...]`，但 `retrieve_multi` 会遍历全部，
  不调用 `_query`；只有 rerank 会取最后一个。

### `_record`

[`graph.py:49`](../../src/babyrag/graph.py#L49)：

```python
return self.llm.record(result, node, note)
```

把 `LLMResult` 转成 `NodeRecord`；缓存命中时 `llm_calls=0`。

### `_hybrid`

[`graph.py:52`](../../src/babyrag/graph.py#L52)：

```python
dense = self.services.dense.search(query, k=10)
bm25 = self.services.bm25.search(query, k=10)
return dense, bm25
```

两路各 10 张，后面 RRF 再截 20。

## 3. `plan`：读工序单

[`graph.py:58`](../../src/babyrag/graph.py#L58)：

```python
mode = state.get("mode", "ladder")
if mode == "ladder":
    rung = state.get("rung", "r0_dense")
    plan = list(RUNGS[rung])
    reason = f"ladder:{rung}"
else:
    rung, reason = adaptive_plan(state["question"], state.get("qtype"))
    plan = list(RUNGS[rung])
return {"plan": plan, "cursor": 0, "route_reason": reason,
        "rung": state.get("rung") or rung,
        "trajectory": [{"node": "plan", "detail": reason}]}
```

两个模式：

- `ladder`：`rung` 是外部传入的固定台阶。
- `adaptive`：`adaptive_plan` 按题型/长度选一档（纯规则，不调 LLM）。

`plan` 返回的 `"rung"` 只是写回 state，当前没有消费者。`run_one` 返回给评测的 `rung`
是它的**入参**（[`graph.py:301`](../../src/babyrag/graph.py#L301)），所以报告按调用时传的
`rung` 分组；adaptive 模式下真正选中的台阶看 `plan` 和 `route_reason`（例如 `type=global`）。

## 4. `dispatch` / `route`

[`graph.py:71`](../../src/babyrag/graph.py#L71)：

```python
def dispatch(self, state):
    return {}
```

[`graph.py:74`](../../src/babyrag/graph.py#L74)：

```python
plan = state.get("plan") or []
cursor = int(state.get("cursor", 0))
return plan[cursor] if cursor < len(plan) else "finalize"
```

`dispatch` 是同步点；`route` 是条件边。每个阶段节点必须返回 `cursor + 1`。

## 5. 检索节点

### 5.1 `retrieve_dense`

[`graph.py:79`](../../src/babyrag/graph.py#L79)：

```python
hits = self.services.dense.search(self._query(state), k=10)
return {"hits": hits, "cursor": state["cursor"] + 1, ...}
```

### 5.2 `retrieve_hybrid`

[`graph.py:84`](../../src/babyrag/graph.py#L84)：

```python
dense, bm25 = self._hybrid(self._query(state))
return {"dense_hits": dense, "bm25_hits": bm25, "cursor": state["cursor"] + 1, ...}
```

注意它**不**直接写 `hits`；下一步 `fuse_rrf` 才融合。

### 5.3 `retrieve_contextual`

[`graph.py:90`](../../src/babyrag/graph.py#L90)：

```python
query = self._query(state)
dense = self.services.contextual_dense.search(query, k=10, source="dense_contextual")
bm25 = self.services.contextual_bm25.search(query, k=10)
return {"dense_hits": dense, "bm25_hits": bm25, ...}
```

和 hybrid 的区别只是换了两套 contextual 集合。

### 5.4 `retrieve_multi`

[`graph.py:98`](../../src/babyrag/graph.py#L98)：

```python
variants = state.get("query_variants") or [state["question"]]
lists = []
for query in variants:
    dense, bm25 = self._hybrid(query)
    lists.append(fuse_rrf(dense, bm25, top_n=10))
return {"multi_hits": lists, ...}
```

每个变体先各自融合成 10 条，再由下一步 `fuse_rrf` 融合所有变体。

## 6. 融合与精排

### 6.1 `fuse_rrf`

[`graph.py:107`](../../src/babyrag/graph.py#L107)：

```python
multi = state.get("multi_hits") or []
dense = state.get("dense_hits") or []
bm25 = state.get("bm25_hits") or []
if multi:
    hits = fuse_rrf(*multi, top_n=20)
else:
    hits = fuse_rrf(dense, bm25, top_n=20)
return {"hits": hits, "cursor": state["cursor"] + 1, ...}
```

两种情况：多查询融合，或单查询的两路融合。

### 6.2 `rerank`

[`graph.py:118`](../../src/babyrag/graph.py#L118)：

```python
hits = rerank_hits(self.services.settings, self._query(state),
                   state.get("hits") or [], top_n=5)
if not hits:
    hits = (state.get("hits") or [])[:5]
return {"hits": hits, ...}
```

如果 `rerank_hits` 返回空（没有 hits，或所有下标都越界），回退到原 hits 前 5。
但**网络错误会直接抛异常**，不会走到这个回退分支；这里的 `if not hits` 只兜底“空结果”。

## 7. 查询变换

### 7.1 `hyde`

[`graph.py:126`](../../src/babyrag/graph.py#L126)：

```python
result = self.llm.invoke(HYDE.format(question=state["question"]),
                         stage="hyde", system="你是一位中文维基百科作者。")
return {"query_variants": [result.text.strip() or state["question"]], ...}
```

模型写一段“假答案”，后续 `retrieve_hybrid` 用 `_query` 取它检索。

### 7.2 `rewrite_multi_query`

[`graph.py:134`](../../src/babyrag/graph.py#L134)：

```python
result = self.llm.invoke(MULTI_QUERY.format(question=state["question"]),
                         stage="multi_query")
variants = [state["question"]] + parse_queries(result.text, 3)
return {"query_variants": variants, ...}
```

`parse_queries`（[`prompts.py:122`](../../src/babyrag/prompts.py#L122)）去掉每行首尾的
空格和 `-•` 符号，最多保留 3 条改写。

### 7.3 `expand_context`

[`graph.py:142`](../../src/babyrag/graph.py#L142)：

```python
extra = self.services.contextual.get(hit["chunk_id"])
if extra:
    hit = dict(hit)
    hit["context"] = extra.get("prefix", "")
    hit["text"] = f"{extra.get('prefix', '')}\n{hit['text']}"
```

给普通 chunk 拼回 contextual 前缀。当前 `RUNGS` 没有用它（r5 直接走 contextual 集合）。

### 7.4 `retrieve_raptor`

[`graph.py:154`](../../src/babyrag/graph.py#L154)：

```python
leaves = self.services.dense.search(query, k=10, source="raptor_leaf")
try:
    summaries = self.services.raptor.search(query, k=10, source="raptor")
except Exception:
    summaries = []
hits = fuse_rrf(leaves, summaries, top_n=20) if summaries else leaves
```

**collapsed tree**：叶子 + 摘要一起查，再融合。只查摘要会丢事实细节。

`except Exception` 会把“摘要 collection 不存在”“SiliconFlow 网络错误”都吞成空摘要，
然后静默退化成只查叶子；如果 r6 效果异常，先确认 `raptor` collection 是否真的建好。

## 调用关系

```text
plan -> dispatch -> route
route -> retrieve_dense / retrieve_hybrid / retrieve_contextual / retrieve_multi
retrieve_hybrid -> fuse_rrf -> rerank
retrieve_contextual -> fuse_rrf -> rerank
retrieve_multi -> fuse_rrf -> rerank
hyde -> retrieve_hybrid
rewrite_multi_query -> retrieve_multi
retrieve_raptor -> rerank
```

## 坑清单

| 坑 | 位置 | 后果 |
|---|---|---|
| 节点忘了 `cursor + 1` | 每个节点 | 图原地打转 |
| `retrieve_hybrid` 直接写 `hits` | graph.py:84 | `fuse_rrf` 拿不到两路 |
| multi-query 只取最后一个 variant | graph.py:98 | 改写没被用上 |
| r5 用 `retrieve_hybrid` | rungs.py | contextual 索引没被查 |
| 把 `if not hits` 当成网络兜底 | graph.py:121 | reranker 抛异常时不会回退 |

## 自测

1. 为什么 `retrieve_hybrid` 不直接写 `hits`？
2. `retrieve_multi` 为什么先对每个 variant 做一次 RRF？
3. `_query` 在 r3 和 r4 下分别返回什么？

### 答案

1. 它要保留 `dense_hits` / `bm25_hits` 两路原始排名，下一步 `fuse_rrf` 才能按 RRF 融合。
2. 每个变体内部先融合 dense+BM25，避免某一路在跨变体融合时权重过大；下一步再融合多个变体。
3. r3：返回 HyDE 的假答案；r4：`retrieve_multi` 不用 `_query`，遍历全部 variants，只有 rerank 会取最后一个。
