---
title: 代码漫游 01 · 地基：config / types / cache
date: 2026-11-24
input_shape: concept
subject: babyrag 的基础模块
---

# 代码漫游 01 · 地基

这套漫游假设你已经看过 [`../README.md`](../README.md) 的通俗版。现在开始读真代码。
顺序不是按文件夹，而是按**依赖方向**：先读别人都依赖的模块。

## 读法

每个函数按四个问题读：

1. **它拿什么**（参数、全局常量）。
2. **它做什么**（逐行）。
3. **它返回什么 / 改了什么**。
4. **谁调用它**。

---

## 1. `config.py`：全项目的地址簿

### 1.1 模块常量

[`config.py:12-25`](../../src/babyrag/config.py#L12) 先定位仓库根，再派生所有路径：

```python
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = ROOT / "eval"
RESULTS = ROOT / "results"
CACHE_DB = DATA / "cache" / "llm.sqlite3"
CORPUS = DATA / "corpus.jsonl"
CHUNKS = DATA / "chunks.jsonl"
```

`parents[2]` 是 `src/babyrag/config.py` 往上三层：`babyrag` → `src` → 仓库根。
这样从任何工作目录运行都能找到 `data/`。

价格和切分参数在 [`config.py:57-64`](../../src/babyrag/config.py#L57)：

```python
PRICE_INPUT = 0.15 / 1e6
PRICE_OUTPUT = 0.60 / 1e6
PRICE_CACHE_READ = 0.003 / 1e6
MAX_CONTEXT_TOKENS = 4000
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
ENCODING = "cl100k_base"
```

**单位坑**：价格是**每 token**，不是每百万 token；`0.15 / 1e6` 已经把“每百万 token 0.15 美元”
换算成单 token 价格。`llm.py` 直接乘 token 数。`PRICE_CACHE_READ` 只在服务端报告
`cached_tokens` 时才非零。

### 1.2 `variant_paths`：三种索引的统一布局

[`config.py:28`](../../src/babyrag/config.py#L28)：

```python
def variant_paths(variant: str) -> dict[str, Path | str]:
    if variant == "masked":
        base, file_suffix, collection_suffix = DATA, ".masked", "_masked"
    elif variant == "public":
        base, file_suffix, collection_suffix = DATA / "public", "", "_public"
    else:
        base, file_suffix, collection_suffix = DATA, "", ""

    return {
        "corpus": base / f"corpus{file_suffix}.jsonl",
        "chunks": base / f"chunks{file_suffix}.jsonl",
        "contextual": base / f"contextual{file_suffix}.jsonl",
        "raptor": base / f"raptor{file_suffix}.jsonl",
        "dense": f"chunks{collection_suffix}",
        "contextual_dense": f"contextual{collection_suffix}",
        "raptor_dense": f"raptor{collection_suffix}",
    }
```

逐行：

- 三个分支只决定 `base`、`file_suffix`、`collection_suffix`。
- 返回值只有一份，所以三种变体的字段**不可能**再分叉。
- `Path | str` 是因为前四个是文件路径，后三个是 Chroma collection 名（字符串）。

**调用者**：`Services.__init__` 把它存成 `self.paths`（[`retrieval.py:96`](../../src/babyrag/retrieval.py#L96)）；
`Settings.paths` 只是转发（[`config.py:86`](../../src/babyrag/config.py#L86)）。

**坑**：`variant="typo"` 会静默落进 `else`，变成 original。要报错的话得显式加 `raise`。

### 1.3 `Settings` 与 `load_settings`

[`config.py:74`](../../src/babyrag/config.py#L74)：

```python
@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    silicon_key: str = ""
    silicon_url: str = DEFAULT_SILICON_URL
    session_id: str = ""
    max_usd: float = 10.0
    variant: str = "original"
    provider: str = "opencode"

    @property
    def paths(self) -> dict[str, Path | str]:
        return variant_paths(self.variant)
```

`frozen=True` 表示构造后不能改。所以 `variant` 是**加载时快照**：
`run_eval.py` 在 `load_settings()` 之前设置 `BABYRAG_VARIANT`，之后改环境变量不会生效。
`provider` 也是这样：它决定聊天模型连 OpenCode 还是 SiliconFlow，不是事后可以切换的标签。

[`config.py:90`](../../src/babyrag/config.py#L90) 的 `load_settings` 先选供应商：

```python
provider = (e.get("BABYRAG_LLM_PROVIDER") or "opencode").strip().lower()
silicon_key = (e.get("SILICONFLOW_API_KEY") or "").strip()
silicon_url = (e.get("SILICONFLOW_BASE_URL") or DEFAULT_SILICON_URL).strip()
session = (e.get("OPENCODE_SESSION_ID") or "").strip() or f"ses_{uuid.uuid4().hex}"

if provider == "siliconflow":
    if not silicon_key:
        raise ConfigError("SILICONFLOW_API_KEY is not set (needed for --provider siliconflow)")
    api_key = silicon_key
    base_url = silicon_url
    model = (e.get("BABYRAG_LLM_MODEL") or "Qwen/Qwen2.5-7B-Instruct").strip()
else:
    provider = "opencode"
    key = (e.get("OPENCODE_GO_API_KEY") or e.get("OPENCODE_API_KEY") or "").strip()
    if not key:
        raise ConfigError("OPENCODE_GO_API_KEY is not set (see .env.example)")
    api_key = key
    base_url = (e.get("OPENCODEGO_BASE_URL") or DEFAULT_BASE_URL).strip()
    model = (e.get("OPENCODEGO_MODEL") or DEFAULT_MODEL).strip()
```

逐段：

- 只认 `BABYRAG_LLM_PROVIDER=siliconflow`；其他值都会被归一成 `opencode`。
- SiliconFlow 分支用 `SILICONFLOW_API_KEY`、`SILICONFLOW_BASE_URL`，模型默认是免费档的 `Qwen/Qwen2.5-7B-Instruct`。
- OpenCode 分支缺 key 直接抛 `ConfigError`，不打印 key；session id 为空就生成 `ses_<uuid>`。
- `variant` 只做字符串透传，不在这里校验；拼错会在 `variant_paths` 里静默回退成 original。
- `usd()`（[`config.py:131`](../../src/babyrag/config.py#L131)）目前**没有调用者**，`llm.py` 内联算了同样的公式。

**读代码时要记住**：`Settings.provider` 同时影响三处：`llm.py` 是否发 `x-opencode-session`、
`invoke` 是否把成本记 0、`finalize` 是否给 `cost["free"]` 打标。只看 `model` 字符串不够。

---

## 2. `types.py`：节点之间传什么

### 2.1 `Chunk`

[`types.py:9`](../../src/babyrag/types.py#L9)：

```python
class Chunk(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    doc_ids: list[str]
    title: str
    text: str
    score: float
    rank: int
    source: str
    level: int
    context: str
```

`total=False`：所有字段都可缺省。读的时候一律 `.get()`，不要假设存在。
`doc_ids` 是给 RAPTOR 摘要用的：一段摘要可能覆盖多篇文档。

### 2.2 `RAGState`

[`types.py:48`](../../src/babyrag/types.py#L48) 是整张流程表。关键在最后两个 reducer 字段：

```python
node_records: Annotated[list[NodeRecord], operator.add]
trajectory: Annotated[list[Step], operator.add]
```

`Annotated[..., operator.add]` 是 LangGraph 的 reducer：节点返回 `[record]` 时，
框架会**追加**到原列表，而不是覆盖。

**坑**：不要在节点里 `state["node_records"].append(...)`，否则 reducer 会再合并一次，
记录翻倍。正确做法是 `return {"node_records": [record]}`。

### 2.3 `add_cost`

[`types.py:93`](../../src/babyrag/types.py#L93)：

```python
def add_cost(records: list[NodeRecord]) -> Cost:
    out: Cost = {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0,
                 "cache_hits": 0, "latency_ms": 0, "usd": 0.0}
    for r in records:
        out["tokens_in"] += int(r.get("tokens_in", 0))
        ...
    return out
```

`finalize` 调它，把每个节点的记录求和成 `state["cost"]`。

---

## 3. `cache.py`：SQLite 备忘录

### 3.1 `_retry`

[`cache.py:21`](../../src/babyrag/cache.py#L21)：

```python
def _retry(fn, *, attempts=8):
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            last = exc
            time.sleep(0.15 * (i + 1))
    raise last
```

只对 `locked` / `busy` 重试；其他 SQLite 错误立刻抛。

### 3.2 `Cache`

[`cache.py:34`](../../src/babyrag/cache.py#L34)：

```python
self._db = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
self._db.execute("PRAGMA busy_timeout=30000")
mode = self._db.execute("PRAGMA journal_mode").fetchone()[0]
if str(mode).lower() != "wal":
    try: self._db.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError: pass
```

- `check_same_thread=False`：放宽 SQLite 的“只能本线程使用”检查。当前评测的 `_env` 每个线程
  会建自己的 `LLMClient` / `Cache`，所以真正共享的是**数据库文件**，不是同一个连接；
  这个参数是防御性放宽。
- `busy_timeout` + `_retry`：多个连接同时写同一个 SQLite 文件时等待并重试，而不是直接失败。
- WAL 只设一次；已经是 WAL 就跳过。

`get` / `put`（[`cache.py:51`](../../src/babyrag/cache.py#L51)、[`cache.py:56`](../../src/babyrag/cache.py#L56)）
都是“包一层 `_retry` 的 SQL”。

### 3.3 `JsonCache`

[`cache.py:69`](../../src/babyrag/cache.py#L69) 是文件版 KV，但**目前没有调用者**。
读代码时不要以为它参与了主流程。

---

## 调用关系

```text
scripts / cli
   └─ load_settings() -> Settings
         └─ Services(settings).paths -> variant_paths()
         └─ LLMClient(settings)
               └─ Cache(CACHE_DB) -> get/put
graph.Runner
   └─ 读写 RAGState（Reducer 追加 node_records/trajectory）
   └─ finalize -> add_cost()
```

## 这一篇的坑清单

| 坑 | 位置 | 后果 |
|---|---|---|
| 价格按每 token | config.py:57 | 忘除 1e6 会放大 100 万倍 |
| variant 是快照 | config.py:74 | 改环境变量要在 load 之前 |
| 未知 variant 静默回退 | config.py:28 | 拼错名字会用 original |
| reducer 字段原地 append | types.py:73 | 记录翻倍 |
| `JsonCache` 未接线 | cache.py:69 | 不要以为它在用 |

## 自测

1. 为什么 `variant_paths` 的返回类型是 `dict[str, Path | str]`？
2. `state["node_records"].append(...)` 为什么会出问题？
3. `Cache` 为什么要 `check_same_thread=False` 和 `busy_timeout`？

### 答案

1. 前四个键是文件路径（`Path`），后三个是 Chroma collection 名（`str`）。
2. `node_records` 有 `operator.add` reducer，框架会把节点返回值再合并一次；原地修改会重复。
3. 评测线程各自建 `Cache`，但多个连接会同时读写同一个 SQLite 文件；`busy_timeout` + `_retry`
   让写操作等待重试。`check_same_thread=False` 则是允许连接跨线程使用的防御性设置。
