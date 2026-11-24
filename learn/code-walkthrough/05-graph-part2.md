---
title: 代码漫游 05 · Graph 下半场：自检、多轮与生成
date: 2026-11-24
input_shape: concept
subject: graph.py 的 grade、agentic、generate、build_graph、run_one
---

# 代码漫游 05 · Graph 下半场

这一篇读 [`graph.py`](../../src/babyrag/graph.py) 的后半段：
`grade`、`agentic`、`generate`、`verify`、`finalize`，以及建图和入口。

## 1. `grade`：CRAG 的自检

[`graph.py:168`](../../src/babyrag/graph.py#L168)。完整逻辑不长：

```python
hits = (state.get("hits") or [])[:6]
if not hits:
    return {"grade": "incorrect", "cursor": state["cursor"] + 1, ...}

chunks = "\n\n".join(f"[{i}] {h['text'][:600]}" for i, h in enumerate(hits, start=1))
result = self.llm.invoke(GRADE.format(question=state["question"], chunks=chunks),
                         stage="grade")
relevant, verdict = parse_relevant(result.text, len(hits))
records = [self._record("grade", result, verdict)]

retries = int(state.get("retries", 0))
if verdict == "incorrect" and retries < 1:
    rewrite = self.llm.invoke(
        f"把这个问题改写成更适合检索的中文查询，只输出一行：\n{state['question']}",
        stage="rewrite")
    query = rewrite.text.strip() or state["question"]
    dense, bm25 = self._hybrid(query)
    new_hits = rerank_hits(self.services.settings, query,
                           fuse_rrf(dense, bm25, top_n=20), top_n=5)
    records.append(self._record("rewrite", rewrite, query))
    return {"hits": new_hits or hits, "grade": verdict, "retries": retries + 1,
            "cursor": state["cursor"] + 1, "node_records": records, ...}

if relevant:
    kept = [hits[i - 1] for i in relevant if 1 <= i <= len(hits)]
    hits = kept
return {"hits": hits, "grade": verdict, "cursor": state["cursor"] + 1,
        "node_records": records, ...}
```

逐段：

1. **只取前 6 条**，每条截 600 字，拼成一个批量 prompt。不是每条一次调用。
2. `parse_relevant`（[`prompts.py:97`](../../src/babyrag/prompts.py#L97)）返回相关序号和
   `correct/ambiguous/incorrect`。解析失败时默认“全保留 + ambiguous”。
3. 只有 `incorrect` 且 `retries < 1` 才重查；**最多一次**。
4. 重查会改写问题、重新 hybrid + rerank，写入新 hits。
5. 否则按 `relevant` 过滤；注意 `relevant` 是 1-based，代码里 `i - 1`。

**为什么把重查放在节点内部？** 如果拆成 `grade → rewrite → retrieve → grade` 的循环，
`plan` 会无限增长；节点内做一次更简单，也好控成本。

## 2. `agentic`：四步 ReAct

[`graph.py:198`](../../src/babyrag/graph.py#L198)。核心：

```python
collected = {}
history = []
for hit in fuse_rrf(*self._hybrid(state["question"]), top_n=5):
    collected.setdefault(hit["chunk_id"], hit)      # baseline 先占位

for step in range(4):
    prompt = AGENTIC.format(question=state["question"],
                            history="\n".join(history) or "(无)")
    result = self.llm.invoke(prompt, stage="agentic", extra_key=str(step))
    text = result.text.strip()
    if "FINAL:" in text:
        break
    query = text.split("SEARCH:", 1)[-1].strip().splitlines()[0] if "SEARCH:" in text else text
    hits = fuse_rrf(*self._hybrid(query), top_n=5)
    for hit in hits:
        collected.setdefault(hit["chunk_id"], hit)
    history.append(f"SEARCH: {query}\nOBSERVATION: " +
                   " ".join(h["text"][:120] for h in hits[:3]))
```

五个设计点：

- **先跑原始问题**：防止模型第一步就改写、把 gold chunk 挤掉；`setdefault` 不覆盖 baseline。
- **`extra_key=str(step)`**：缓存键按步区分，同一题重跑每步仍能命中。
- **最多 4 步**、`FINAL:` 提前结束；没有收集到任何 hit 时回退一次 hybrid。
- `FINAL:` 只用来结束搜索；它后面的文字**不会**直接变成答案，最终答案仍由 `generate`
  基于收集到的 `hits` 重写。
- `history` 里的 `OBSERVATION` 不是完整 chunk，而是前 3 条命中的前 120 字；
  这样下一轮 prompt 不会因为历史太长而爆预算。

## 3. `generate`：写答案

[`graph.py:230`](../../src/babyrag/graph.py#L230)：

```python
hits = (state.get("hits") or [])[:5]
context, kept = build_context(hits, services=self.services,
                              budget=MAX_CONTEXT_TOKENS)
if not kept:
    return {"answer": "无法回答", "abstain": True, "hits": [], ...}

result = self.llm.invoke(GENERATE.format(question=state["question"], context=context),
                         stage="generate", system=SYSTEM_GROUNDED)
answer = result.text.strip()
# A refusal is a refusal even if the model appends an explanation.
abstain = answer.startswith("无法回答")
return {"answer": answer, "abstain": abstain, "hits": kept,
        "citations": extract_citations(answer, kept), ...}
```

- **top-5 是硬规则**，不因上游返回 8 条而改变。
- `kept` 是实际进上下文的列表；引用编号基于它。
- `abstain` 只看**开头**：只要答案以“无法回答”开头就算弃答，后面附加解释也算。
  不是“整段等于四个字”，也不是“长度小于 20 字”。

## 4. `verify` / `finalize`

[`graph.py:250`](../../src/babyrag/graph.py#L250)：

```python
answer = state.get("answer") or ""
abstain = state.get("abstain", False) or answer.strip() == "无法回答"
return {"abstain": abstain, ...}
```

它只做最终确认，没有引用合法性检查。`finalize`（[`graph.py:256`](../../src/babyrag/graph.py#L256)）：

```python
records = state.get("node_records") or []
cost = add_cost(records)
cost["free"] = self.services.settings.provider == "siliconflow"
return {"cost": cost, ...}
```

`free` 是给评测看的：SiliconFlow 免费档的 `usd` 是 0，`summary` 算冷启动成本时也会跳过它，
避免把免费模型和 DeepSeek 主榜混在一起。

## 5. `build_graph`：把节点连起来

[`graph.py:264`](../../src/babyrag/graph.py#L264)：

```python
runner = Runner(services, llm)
builder = StateGraph(RAGState)
builder.add_node("plan", runner.plan)
builder.add_node("dispatch", runner.dispatch)
for name in STAGES:
    builder.add_node(name, getattr(runner, name))
builder.add_node("finalize", runner.finalize)

builder.add_edge(START, "plan")
builder.add_edge("plan", "dispatch")
mapping = {name: name for name in STAGES}
mapping["finalize"] = "finalize"
builder.add_conditional_edges("dispatch", runner.route, mapping)
for name in STAGES:
    builder.add_edge(name, "dispatch")
builder.add_edge("finalize", END)
return builder.compile()
```

- `getattr(runner, name)` 要求每个 `STAGES` 名字都有同名方法；漏一个会在建图时报错。
- 条件边的 `mapping` 必须覆盖 `route` 可能返回的所有值。
- 每个阶段都有一条静态边回 `dispatch`，所以 `cursor` 是唯一推进机制。

## 6. `run_one`：入口

[`graph.py:289`](../../src/babyrag/graph.py#L289)：

```python
state = {
    "qid": qid, "question": question, "rung": rung, "mode": mode,
    "qtype": qtype, "retries": 0, "node_records": [], "trajectory": [],
}
result = graph.invoke(state, config={"recursion_limit": 60})

cost = dict(result.get("cost") or {})
cost["latency_ms"] = elapsed
return {
    "qid": qid, "rung": rung, "question": question,
    "answer": result.get("answer", ""), "abstain": bool(result.get("abstain", False)),
    "citations": result.get("citations", []),
    "hits": [{"chunk_id": ..., "doc_id": ..., "doc_ids": ..., "text": ...,
              "score": ..., "source": ...}],
    "plan": result.get("plan", []),
    "route_reason": result.get("route_reason", ""),
    "cost": cost,
    "node_records": result.get("node_records", []),
}
```

- `recursion_limit=60` 防死循环。
- `cost["latency_ms"]` 用墙钟覆盖节点累计值，包含排队/调度时间。
- `hits` 保留 `text` 和 `doc_ids`，因为评测要算 evidence recall。

## 调用关系

```text
dispatch -> grade (r7) -> generate
dispatch -> agentic (r8) -> verify -> generate
generate -> build_context -> extract_citations -> finalize
build_graph -> compile -> run_one -> graph.invoke
```

## 坑清单

改这里时最容易踩的坑：

| 易错点 | 位置 | 后果 |
|---|---|---|
| 把 6 条 chunk 拆成 6 次 grade 调用 | graph.py:173 | 成本 ×6，且更容易触发限流 |
| 把重查拆成图里的循环 | graph.py:179 | `plan` 越来越长，成本和步数都难控 |
| agentic 用赋值覆盖 baseline | graph.py:205 | 原始查询命中的 gold chunk 被改写挤掉 |
| `generate` 忘了再截 top-5 | graph.py:232 | 不同 rung 的上下文预算不再公平 |
| `run_one` 不保留 `hits.text` | graph.py:304 | 评测无法算 evidence recall |

## 自测

1. `grade` 最多调用几次 LLM？分别在什么条件下？
2. 为什么 `agentic` 要先跑一次原始问题？
3. `build_graph` 里如果 `STAGES` 多了一个名字但没有对应方法，会怎样？

### 答案

1. 正常 1 次（grade）；如果 verdict=incorrect 且未重试，再加 1 次改写，共 2 次（重查本身不调 LLM）。
2. 防止模型第一步就改写查询把 gold chunk 挤掉；baseline 先占住前 5，后续 `setdefault` 只补充。
3. `getattr(runner, name)` 会抛 `AttributeError`，建图失败。
