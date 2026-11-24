---
title: 代码漫游 02 · 模型调用与缓存
date: 2026-11-24
input_shape: concept
subject: llm.py 逐函数阅读
---

# 代码漫游 02 · 模型调用与缓存

这一篇只读 [`llm.py`](../../src/babyrag/llm.py)。它很短，但每行都和钱、重试、缓存有关。

## 0. 先看结构

```
_key()            # 算缓存指纹
LLMResult         # 一次调用的结果（text + token + usd）
LLMClient
  __init__        # 建 ChatOpenAI、SQLite 缓存
  invoke          # 缓存 -> 调用 -> 计费 -> 落盘
  _invoke_with_backoff
  record          # LLMResult -> NodeRecord
  budget_left
_message_text     # 从 LangChain 消息里抠文本
_usage            # usage_metadata -> response_metadata -> tiktoken
build_embeddings  # bge-m3
rerank            # SiliconFlow /v1/rerank
```

## 1. 指纹：`_key`

[`llm.py:29`](../../src/babyrag/llm.py#L29)：

```python
def _key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
```

调用方传六个部分（[`llm.py:70`](../../src/babyrag/llm.py#L70)）：

```python
key = _key(stage, prompt_version, self.model, system or "", prompt, extra_key)
```

所以下面任何一项变化，都会让缓存失效：

- 阶段名（`generate` / `hyde` / `grade` ...）
- 提示词版本
- 模型名
- system prompt
- 用户 prompt（包含检索到的上下文）
- 额外 key（agentic 的步数）

**接线坑**：`prompts.py:8` 定义了 `PROMPT_VERSION = "v1"`，但目前没有任何节点 import 它；
`invoke` 的默认参数是 `prompt_version="v1"`。所以改这个常量不会自动改变缓存版本，
改了 prompt 文本则一定会改变 key（因为完整 prompt 也在 key 里）。

## 2. `LLMResult`

[`llm.py:34`](../../src/babyrag/llm.py#L34)：

```python
@dataclass
class LLMResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    cached: bool = False
    latency_ms: int = 0
```

缓存命中时 `cached=True`、`usd=0`，但 `tokens_in/out` 仍然保留——这是为了算“冷启动成本”。

## 3. 构造：`LLMClient.__init__`

[`llm.py:46`](../../src/babyrag/llm.py#L46)：

```python
self.cache = Cache(CACHE_DB)
headers = ({"x-opencode-session": settings.session_id}
           if settings.provider == "opencode" else None)
self._llm = ChatOpenAI(
    model=settings.model,
    api_key=settings.api_key,
    base_url=settings.base_url,
    temperature=temperature,
    max_tokens=max_output_tokens,
    timeout=120,
    max_retries=2,
    default_headers=headers,
)
```

三个重点：

- 只有 `provider == "opencode"` 才发 `x-opencode-session`；这是 OpenCode Go 的硬要求，缺了返回 400。
  走 SiliconFlow 时 `headers=None`，不需要这个请求头。
- `max_retries=2` 是 SDK 自带重试；外面还有 `_invoke_with_backoff` 的 5 次。两者会叠加，最坏情况调用次数比预期多，但能扛住网关抖动。
- `provider` 还决定计费：SiliconFlow 分支把 `cost` 记 0（见 4.3），不是只改一个显示字段。

## 4. 核心：`invoke`

[`llm.py:67`](../../src/babyrag/llm.py#L67)。按执行顺序拆：

### 4.1 先查缓存

```python
if use_cache:
    hit = self.cache.get(key)
    if hit is not None:
        try:
            payload = json.loads(hit)
        except json.JSONDecodeError:
            payload = {"text": hit}
        return LLMResult(
            text=payload.get("text", ""),
            tokens_in=int(payload.get("tokens_in", 0)),
            tokens_out=int(payload.get("tokens_out", 0)),
            cached=True,
        )
```

`payload = {"text": hit}` 是兼容早期只存纯文本的缓存。新缓存都是 JSON。
命中时 `latency_ms` 保持默认 0、`usd=0`，但 `tokens_in/out` 从缓存里读回来，
所以报告仍能估算“如果没有缓存要花多少”。

### 4.2 真的调用

```python
messages = []
if system:
    messages.append(SystemMessage(content=system))
messages.append(HumanMessage(content=prompt))
response = self._invoke_with_backoff(messages)
```

### 4.3 算钱

[`llm.py:92`](../../src/babyrag/llm.py#L92)：

```python
tin, tout, cached_in = _usage(response, messages, text)
if self.settings.provider == "siliconflow":
    cost = 0.0  # free-tier chat model on SiliconFlow
else:
    cost = (tin * PRICE_INPUT + tout * PRICE_OUTPUT + cached_in * PRICE_CACHE_READ)
self.spent_usd += cost
```

**`provider` 分支**：如果聊天模型也走 SiliconFlow 免费档，成本直接记 0。
默认 `provider="opencode"`，走 DeepSeek 计价。

### 4.4 写缓存 + 记账

```python
self.cache.put(key, json.dumps(
    {"text": text, "tokens_in": tin, "tokens_out": tout},
    ensure_ascii=False), self.model)
...
with COST_LOG.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

`results/cost.jsonl` 是 append-only；每次真实调用一行。缓存命中不写这一行（因为没花钱）。
只有 `use_cache=True` 才读写缓存；传 `use_cache=False` 时会完全跳过这两步。

## 5. 重试：`_invoke_with_backoff`

[`llm.py:114`](../../src/babyrag/llm.py#L114)：

```python
for attempt in range(5):
    try:
        return self._llm.invoke(messages)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        retryable = any(s in text for s in ("429", "Rate limit", "rate_limit",
                                            "500", "502", "503", "504", "timeout",
                                            "Timeout", "overloaded",
                                            "Connection error", "Connection"))
        if not retryable or attempt == 4:
            raise
        time.sleep(min(30.0, 2.0 * (2 ** attempt)) + random.random())
```

为什么按字符串匹配，而不是按异常类型？因为不同 httpx/网关版本抛的类型不一致。
代价是可能误判，但比漏掉 429 好。

## 6. token 从哪来：`_usage`

[`llm.py:158`](../../src/babyrag/llm.py#L158) 是三级回退：

```python
meta = getattr(response, "usage_metadata", None) or {}
...
if tin and tout:
    return tin, tout, cached

raw = getattr(response, "response_metadata", None) or {}
usage = raw.get("token_usage") or raw.get("usage") or {}
...
if tin or tout:
    return tin, tout, cached

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tin = sum(len(enc.encode(m.content)) for m in messages)
tout = len(enc.encode(text))
```

- 优先级：LangChain 标准字段 → 网关原始字段 → 本地估算。
- 本地估算会把 system + user 都算进去，但不含模型内部开销，所以偏低。

`_message_text`（[`llm.py:145`](../../src/babyrag/llm.py#L145)）处理 `content` 不是字符串的情况：
有些网关返回 `[{"type": "text", "text": "..."}]` 这样的 block 列表，它只拼 `text` 字段，
避免把 JSON 结构直接写进答案或缓存。

## 7. `record` / `budget_left`

[`llm.py:134`](../../src/babyrag/llm.py#L134)：

```python
llm_calls=0 if result.cached else 1,
cache_hits=1 if result.cached else 0,
```

缓存命中不会被算成一次 LLM 调用。`budget_left`（[`llm.py:141`](../../src/babyrag/llm.py#L141)）
当前没有被图调用，是一个预留接口。

## 8. Embedding 与 rerank

[`llm.py:185`](../../src/babyrag/llm.py#L185)：

```python
OpenAIEmbeddings(
    model=EMBED_MODEL,
    base_url=settings.silicon_url,
    api_key=settings.silicon_key,
    check_embedding_ctx_length=False,
    chunk_size=batch_size,
)
```

`check_embedding_ctx_length=False` 必须：否则 LangChain 会在本地把文本转 token id，
而 SiliconFlow 只接受原始文本。

[`llm.py:197`](../../src/babyrag/llm.py#L197) 的 `rerank` 直接 POST：

```python
if not settings.silicon_key or not documents:
    return list(enumerate([0.0] * len(documents)))[:top_n]
response = httpx.post(
    f"{settings.silicon_url}/rerank",
    headers={"Authorization": f"Bearer {settings.silicon_key}"},
    json={"model": RERANK_MODEL, "query": query,
          "documents": documents, "top_n": min(top_n, len(documents))},
    timeout=timeout,
)
response.raise_for_status()
body = response.json()
return [(int(r["index"]), float(r["relevance_score"])) for r in body["results"]]
```

返回的 `index` 是**输入列表下标**，调用方必须用它映射回原 Chunk。
没有 key 时，它返回保序的 `(0, 0.0), (1, 0.0), ...`，精排会静默退化成“保持原顺序”；
没有文档时返回 `[]`，`rerank_hits` 也返回空，`Runner.rerank` 再回退到原 hits 前 5。
两种都不报错。

## 调用关系

```text
graph.Runner.hyde/rewrite_multi_query/grade/agentic/generate
  -> LLMClient.invoke()
       -> Cache.get/put
       -> ChatOpenAI.invoke
       -> _usage -> 计费
       -> results/cost.jsonl

eval.judge / judge_ab
  -> LLMClient.invoke(stage="judge"/"judge_swap")

retrieval.build_context
  -> Services.count_tokens（不是 LLM）
```

## 坑清单

| 坑 | 位置 | 后果 |
|---|---|---|
| OpenCode 分支缺 `x-opencode-session` | llm.py:53 | 网关 400 |
| 以为 `PROMPT_VERSION` 已接线 | prompts.py:8 | 改常量不会换缓存版本 |
| 自己另算一次调用时忽略 `result.cached` | llm.py:134 | `llm_calls` 虚高 |
| `provider` 配错 | config.py:93 / llm.py:93 | 免费档被按美元计费，或反之 |
| rerank 下标没映射回原列表 | retrieval.py:199 | 排序结果对错卡片 |

## 自测

1. 同一道题跑第二遍，`cost.jsonl` 会不会多一行？为什么？
2. 为什么 `invoke` 缓存命中时还要返回 token 数？
3. `_usage` 的三级回退分别是什么？

### 答案

1. 不会。缓存命中直接 return，不写成本日志；只有真实调用才 append。
2. 为了在 `summary.json` 里计算“如果没有缓存要花多少”（冷启动成本），以及区分 `llm_calls` / `cache_hits`。
3. `usage_metadata` → `response_metadata` 的 `token_usage`/`usage` → tiktoken 本地估算。
