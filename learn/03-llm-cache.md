---
title: 模型与缓存
date: 2026-11-23
input_shape: concept
subject: llm.py 与 cache.py
---

# 模型与缓存

这一层只做三件事：**发请求、算钱、别重复发**。

## LLMClient 的构造

[llm.py:43](../src/babyrag/llm.py#L43)：

```python
self._llm = ChatOpenAI(
    model=settings.model,
    api_key=settings.api_key,
    base_url=settings.base_url,
    temperature=0.0,
    max_tokens=max_output_tokens,
    timeout=120,
    max_retries=2,
    default_headers={"x-opencode-session": settings.session_id},
)
```

两个关键点：

- **`x-opencode-session` 必须存在**。OpenCode Go 缺这个头会返回 400
  （`Request is missing x-opencode-session ...`）。`config.py` 保证它非空。
- 同一个进程复用同一个 session，才能命中网关的提示词缓存。

## invoke 的完整流程

[llm.py:65](../src/babyrag/llm.py#L65)：

```mermaid
flowchart TD
  A[invoke] --> B[sha256(stage, prompt_version, model, system, prompt, extra)]
  B --> C{cache.get}
  C -->|命中| D[返回 text + 原始 tokens, cached=True, usd=0]
  C -->|未命中| E[ChatOpenAI.invoke]
  E --> F[usage_metadata → response_metadata → tiktoken]
  F --> G[cost = tin*IN + tout*OUT + cached_in*CACHE]
  G --> H[cache.put JSON]
  H --> I[append results/cost.jsonl]
  I --> J[返回 LLMResult]
```

关键代码：

- 缓存键：[llm.py:68](../src/babyrag/llm.py#L68)，包含 `prompt_version`；改提示词时递增版本，旧缓存自然失效。
- 缓存命中：[llm.py:69](../src/babyrag/llm.py#L69)，返回**原始 token 数**但 `usd=0`，这样汇总时还能算“冷启动成本”。
- 计费：[llm.py:91](../src/babyrag/llm.py#L91)，`PRICE_INPUT` 等已经是每 token 价。
- 落盘：[llm.py:93](../src/babyrag/llm.py#L93) 写 SQLite，[llm.py:97](../src/babyrag/llm.py#L97) 追加 `results/cost.jsonl`。

## 重试

[llm.py:109](../src/babyrag/llm.py#L109) 最多 5 次，指数退避 + 随机抖动，上限 30 秒。
哪些错误重试：

```python
("429", "Rate limit", "rate_limit", "500", "502", "503", "504",
 "timeout", "Timeout", "overloaded", "Connection error", "Connection")
```

这是按**错误文本**匹配，不是按异常类型；原因是不同网关/httpx 版本抛的类型不一致。
代价是可能误判，但比漏掉 429 好。

## token 用量从哪来

[llm.py:153](../src/babyrag/llm.py#L153) 三级回退：

1. `AIMessage.usage_metadata`（LangChain 标准字段）；
2. `response_metadata["token_usage"]` / `["usage"]`（OpenAI 兼容网关）；
3. `tiktoken` 本地估算（最后兜底）。

`cached_in` 只从 `input_token_details.cache_read` 或
`prompt_tokens_details.cached_tokens` 取；没有就是 0。

## 缓存实现

[cache.py:34](../src/babyrag/cache.py#L34)：

- `sqlite3.connect(..., timeout=30, check_same_thread=False)`；
- `PRAGMA busy_timeout=30000`；
- 首次打开尝试 `journal_mode=WAL`；
- `_retry`（[cache.py:21](../src/babyrag/cache.py#L21)）对 `database is locked` / `busy` 重试 8 次。

这是为了评测时多个线程同时读写同一个库。`get`/`put` 都包了 `_retry`
（[cache.py:51](../src/babyrag/cache.py#L51)、[cache.py:56](../src/babyrag/cache.py#L56)）。

## Embedding 与 rerank

[llm.py:180](../src/babyrag/llm.py#L180)：

```python
OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url=settings.silicon_url,
    api_key=settings.silicon_key,
    check_embedding_ctx_length=False,
    chunk_size=batch_size,
)
```

`check_embedding_ctx_length=False` 是必须的：LangChain 默认会在本地把文本转成 token id
再发给 OpenAI；SiliconFlow 不是 OpenAI，只接受原始文本，所以必须关掉。

[llm.py:192](../src/babyrag/llm.py#L192) 的 `rerank` 直接 `httpx.post /v1/rerank`，
返回 `[(index, relevance_score)]`。索引是**输入列表的下标**，`rerank_hits` 靠它映射回原 Chunk。

## 成本口径

| 项 | 公式 |
|---|---|
| 输入 | `tokens_in × $0.15 / 1M` |
| 输出 | `tokens_out × $0.60 / 1M` |
| 缓存读 | `cached_in × $0.003 / 1M` |
| 缓存命中（本地） | `$0`，但保留 tokens 用于“冷启动成本” |

`LLMClient.spent_usd` 只累计**真实花费**；缓存命中不增加。评测里的 `usd_cold_*`
则按 tokens 重新算，不受缓存影响（[eval.py:211](../src/babyrag/eval.py#L211)）。

## Check yourself

1. 没有 `x-opencode-session` 会发生什么？为什么不是鉴权失败？
2. 缓存命中时为什么还要返回 `tokens_in/tokens_out`？
3. 为什么 `check_embedding_ctx_length` 必须设为 `False`？

### Answers

1. 网关返回 HTTP 400，消息是缺少会话头；鉴权先通过了，所以看起来像请求体/模型问题。`config.py` 生成 `ses_<uuid>` 兜底。
2. 为了让 `summary.json` 能算“如果没有缓存要花多少钱”，以及让 `llm_calls/cache_hits` 统计正确。命中本身 `usd=0`。
3. LangChain 默认会用 tiktoken 把文本转 token id 发给 OpenAI 兼容端点；SiliconFlow 的 bge-m3 只接受文本，收到 token id 会报错或得到错误向量。
