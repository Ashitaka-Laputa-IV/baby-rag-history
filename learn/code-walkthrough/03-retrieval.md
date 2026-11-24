---
title: 代码漫游 03 · 检索层
date: 2026-11-24
input_shape: concept
subject: retrieval.py 逐函数阅读
---

# 代码漫游 03 · 检索层

这一篇读 [`retrieval.py`](../../src/babyrag/retrieval.py)。它不调用生成模型，只负责“找卡片”和“拼上下文”。

## 0. 结构

```
read_jsonl / tokenize
DenseRetriever        # Chroma 向量检索
BM25Retriever         # 关键词检索
Services              # 懒加载容器
get_services          # 未接线
fuse_rrf              # 排名融合
rerank_hits           # 精排
build_context         # 拼上下文
extract_citations     # 从答案里抽 [n]
```

## 1. 小工具

[`retrieval.py:24`](../../src/babyrag/retrieval.py#L24)：

```python
def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]
```

用 `utf-8-sig` 兼容带 BOM 的文件。

[`retrieval.py:29`](../../src/babyrag/retrieval.py#L29)：

```python
def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if TOKEN_RE.search(t)]
```

`jieba.lcut` 切中文，正则过滤纯标点/空白。索引和查询必须用同一个函数。

## 2. `DenseRetriever`

### 2.1 构造

[`retrieval.py:33`](../../src/babyrag/retrieval.py#L33)：

```python
with _INIT_LOCK:
    self.store = Chroma(
        collection_name=collection,
        embedding_function=build_embeddings(settings),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
```

- `_INIT_LOCK`（[`retrieval.py:21`](../../src/babyrag/retrieval.py#L21)）串行化 Chroma 初始化。
  并发初始化同一个 persist 目录会报 tenant/bindings 错误。
- `collection_metadata` 只在**新建** collection 时生效；已存在的 collection 不会被改度量。

### 2.2 搜索

[`retrieval.py:46`](../../src/babyrag/retrieval.py#L46)：

```python
pairs = self.store.similarity_search_with_score(query, k=k)
for rank, (doc, score) in enumerate(pairs, start=1):
    meta = doc.metadata or {}
    doc_id = str(meta.get("doc_id", ""))
    raw_ids = str(meta.get("doc_ids", "") or "")
    doc_ids = [x for x in raw_ids.split("|") if x] or ([doc_id] if doc_id else [])
```

- `score` 是**距离**（cosine 越小越近），不是相似度。
- `doc_ids` 是 `|` 分隔的字符串；RAPTOR 摘要会覆盖多篇文档。
- `level` 区分叶子和摘要。

## 3. `BM25Retriever`

[`retrieval.py:67`](../../src/babyrag/retrieval.py#L67)：

```python
self._tokens = [tokenize(r[text_field]) for r in rows]
self._bm25 = BM25Okapi(self._tokens)
```

[`retrieval.py:75`](../../src/babyrag/retrieval.py#L75)：

```python
scores = self._bm25.get_scores(tokenize(query))
order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
for rank, i in enumerate(order, start=1):
    row = self.rows[i]
    if scores[i] <= 0:
        continue
```

- 索引在内存里重建，没有落盘；4,740 个 chunk 很快。
- 零分文档被丢掉，避免把完全不相关的卡片塞进 RRF。
- 注意顺序：先截前 `k` 个下标，再过滤零分，所以返回条数**可能少于 `k`**。

## 4. `Services`：懒加载

[`retrieval.py:93`](../../src/babyrag/retrieval.py#L93)。它只持有引用，第一次访问才建：

```python
@property
def dense(self):
    if self._dense is None:
        self._dense = DenseRetriever(self.settings, self.paths["dense"])
    return self._dense
```

| 属性 | 依赖 |
|---|---|
| `chunks` | 读 `paths["chunks"]` |
| `contextual` | 读 `paths["contextual"]` |
| `dense` | `paths["dense"]` |
| `contextual_dense` | `paths["contextual_dense"]` |
| `raptor` | `paths["raptor_dense"]` |
| `bm25` | 从 `chunks` 建 |
| `contextual_bm25` | 从 `contextual` 建 |

`get_services`（[`retrieval.py:163`](../../src/babyrag/retrieval.py#L163)）是一个**未接线**的全局单例；
实际用的是 `eval._env` 里的线程本地 `Services`。读代码时不要被它误导。

## 5. `fuse_rrf`

[`retrieval.py:170`](../../src/babyrag/retrieval.py#L170)：

```python
for hits in lists:
    for rank, hit in enumerate(hits, start=1):
        cid = hit["chunk_id"]
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in best or hit.get("score", 0) > best[cid].get("score", 0):
            best[cid] = hit

ordered = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_n]
for rank, cid in enumerate(ordered, start=1):
    hit = dict(best[cid])
    hit["rank"] = rank
    hit["score"] = scores[cid]
    hit["source"] = "rrf"
```

- `k=60`（[`retrieval.py:19`](../../src/babyrag/retrieval.py#L19)）。
- `if not cid: continue` 丢弃空 id；否则多条坏卡片会共用一个空 key。
- 只看名次，避免 dense 距离和 BM25 分数尺度不同。
- `best[cid]` 保留**原始分数最高**的那份 Chunk，用于 `text/doc_ids`；
  最终 `score` 被替换成 RRF 分。
- 这里有个语义不一致：dense 的 `score` 是距离（越小越好），所以对 dense 重复命中，
  “更高分”实际是**更远**的那份。因为同一 `chunk_id` 的正文相同，最终只影响被覆盖的
  `score`，不影响 `text`；但读这段时不要把它理解成“挑最相关的”。
- 输出 `source="rrf"`。

## 6. `rerank_hits`

[`retrieval.py:192`](../../src/babyrag/retrieval.py#L192)：

```python
pairs = rerank(settings, query, [h["text"] for h in hits], top_n=top_n)
for rank, (index, score) in enumerate(pairs, start=1):
    hit = dict(hits[index])
    hit["score"] = float(score)
    hit["rank"] = rank
    hit["source"] = "rerank"
```

`index` 是**输入列表下标**，不是 `chunk_id`。如果 reranker 返回乱序或越界，
`if not (0 <= index < len(hits)): continue` 会跳过。

`llm.rerank` 在没配 `SILICONFLOW_API_KEY` 时不会发请求，而是返回
`[(0, 0.0), (1, 0.0), ...]` 的前 `top_n` 个下标；这时精排退化成“保持原顺序”，
不会报错，也不会有真实相关性分数。

## 7. `build_context`

[`retrieval.py:209`](../../src/babyrag/retrieval.py#L209)：

```python
for hit in hits:
    title = hit.get("title", "")
    body = hit["text"]
    piece = f"[{len(kept) + 1}] （{title}）{body}" if title else f"[{len(kept) + 1}] {body}"
    cost = services.count_tokens(piece)
    if used + cost > budget:
        continue
    kept.append(hit)
    pieces.append(piece)
    used += cost
```

- 编号基于 `kept`，不是原始 hits。
- 超预算的 chunk 被**跳过**，后面的小 chunk 仍可能进来。
- `title` 会拼进上下文；masked 变体只替换了 chunk 正文里的标题，metadata 的 title 仍会出现在这里。

## 8. `extract_citations`

[`retrieval.py:227`](../../src/babyrag/retrieval.py#L227)：

```python
for match in re.finditer(r"\[(\d{1,2})\]", answer):
    index = int(match.group(1)) - 1
    if 0 <= index < len(hits):
        cid = hits[index]["chunk_id"]
```

引用编号是 1-based；越界的 `[99]` 被丢掉。它只检查“引用的 chunk 在不在 hits 里”，
不检查“这个 chunk 是否真的支持这句话”。

## 调用关系

```text
graph.Runner
  retrieve_dense      -> Services.dense.search
  retrieve_hybrid     -> Services.dense.search + Services.bm25.search
  retrieve_contextual -> Services.contextual_dense + contextual_bm25
  retrieve_raptor     -> Services.dense + Services.raptor
  fuse_rrf            -> retrieval.fuse_rrf
  rerank              -> retrieval.rerank_hits
  generate            -> retrieval.build_context + extract_citations
```

## 坑清单

| 坑 | 位置 | 后果 |
|---|---|---|
| 把 Chroma score 当相似度 | retrieval.py:59 | 排序方向反了 |
| BM25 零分未过滤 | retrieval.py:82 | 无关卡片进 RRF |
| rerank 下标映射错 | retrieval.py:199 | 排序结果对错卡片 |
| 超预算直接 break | retrieval.py:220 | 丢掉后面更小的可用 chunk |
| 以为 masked title 完全消失 | retrieval.py:215 | 上下文仍可能出现标题 |

## 自测

1. 为什么 RRF 用 `1/(60+rank)` 而不是原始分数？
2. `build_context` 超预算时为什么是 `continue` 而不是 `break`？
3. `rerank_hits` 的 `index` 和 `chunk_id` 有什么区别？

### 答案

1. 两路分数尺度不同；用名次可避免某一路支配。60 是平滑常数。
2. 后面可能有更小的 chunk 仍能装进预算；`break` 会浪费剩余空间。
3. `index` 是 reranker 输入列表的位置（0-based），`chunk_id` 是卡片主键；必须用 index 映射回原 Chunk。
