---
title: 配置与类型
date: 2026-11-23
input_shape: concept
subject: config.py 与 types.py
---

# 配置与类型

`config.py` 管“在哪里、多少钱、用哪个索引”；`types.py` 管“在节点之间传什么”。

## config.py：路径、价格、环境

### 目录常量

[config.py:12](../src/babyrag/config.py#L12) 用 `Path(__file__).resolve().parents[2]` 定位仓库根，
所以从任何工作目录运行都能找到 `data/`、`eval/`、`results/`。

```python
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = ROOT / "eval"
RESULTS = ROOT / "results"
CACHE_DB = DATA / "cache" / "llm.sqlite3"
```

### 三套索引变体

[variant_paths:28](../src/babyrag/config.py#L28) 返回同一组键，换的是文件和 Chroma collection：

| 键 | original | masked | public |
|---|---|---|---|
| `corpus` | `data/corpus.jsonl` | `corpus.masked.jsonl` | `data/public/corpus.jsonl` |
| `chunks` | `chunks.jsonl` | `chunks.masked.jsonl` | `public/chunks.jsonl` |
| `contextual` | `contextual.jsonl` | `contextual.masked.jsonl` | `public/contextual.jsonl` |
| `raptor` | `raptor.jsonl` | `raptor.masked.jsonl` | `public/raptor.jsonl` |
| `dense` | `chunks` | `chunks_masked` | `chunks_public` |
| `contextual_dense` | `contextual` | `contextual_masked` | `contextual_public` |
| `raptor_dense` | `raptor` | `raptor_masked` | `raptor_public` |

`Settings.paths` 只是 `variant_paths(self.variant)` 的转发
（[config.py:94](../src/babyrag/config.py#L94)），`Services` 再按这些键懒加载。

### 价格与常量

[config.py:61](../src/babyrag/config.py#L61)：

```python
PRICE_INPUT = 0.15 / 1e6      # $/token
PRICE_OUTPUT = 0.60 / 1e6
PRICE_CACHE_READ = 0.003 / 1e6
MAX_CONTEXT_TOKENS = 4000
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
```

**注意单位**：这里存的是**每 token** 的美元价，不是每百万 token。`llm.py` 直接乘 token 数。

### Settings 与 load_settings

[config.py:83](../src/babyrag/config.py#L83) 是 `@dataclass(frozen=True)`，构造后不可变。

[config.py:99](../src/babyrag/config.py#L99) 做了四件事：

1. `load_dotenv(ROOT / ".env")`；
2. 读 `OPENCODE_GO_API_KEY`，回退 `OPENCODE_API_KEY`，都没有就抛 `ConfigError`；
3. 读 `OPENCODE_SESSION_ID`，没有就生成 `ses_<uuid>`；
4. 读 `BABYRAG_VARIANT` 决定索引变体。

**坑**：`variant` 是加载时快照。`run_eval.py` 在 `load_settings()` 之前设置
`os.environ["BABYRAG_VARIANT"]`；如果你在 Python 里先建了 `Settings` 再改环境变量，不会生效。

### 一个未使用的小函数

[config.py:123](../src/babyrag/config.py#L123) 的 `usd()` 目前没有被调用；
`llm.py` 在 `invoke` 里内联算了同样的价格。读代码时不要以为调用链会经过它。

## types.py：节点之间传什么

### Chunk

[types.py:9](../src/babyrag/types.py#L9)：检索结果的最小单位。

```python
class Chunk(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    doc_ids: list[str]   # RAPTOR 摘要覆盖多个文档时用
    title: str
    text: str
    score: float
    rank: int
    source: str          # dense | bm25 | rrf | rerank | raptor ...
    level: int           # RAPTOR 层级
    context: str         # contextual prefix
```

`total=False` 意味着字段都可缺省；读的时候一律用 `.get()`，不要假设一定存在。

### RAGState

[types.py:47](../src/babyrag/types.py#L47) 按阶段分组：

| 组 | 字段 | 谁写 |
|---|---|---|
| 输入/计划 | `qid`, `question`, `mode`, `rung`, `plan`, `cursor`, `route_reason` | `run_one` / `plan` |
| 查询 | `qtype`, `query_variants` | `hyde` / `rewrite_multi_query` |
| 检索 | `dense_hits`, `bm25_hits`, `multi_hits`, `hits` | 各 retrieve / fuse |
| 生成 | `answer`, `citations` | `generate` |
| 控制 | `grade`, `retries`, `abstain` | `grade` / `generate` / `verify` |
| 观测 | `node_records`, `trajectory`, `cost` | 所有节点 / `finalize` |

两个 reducer 字段：

```python
node_records: Annotated[list[NodeRecord], operator.add]
trajectory: Annotated[list[Step], operator.add]
```

节点返回 `[record]` 这种**增量**，LangGraph 自动 append。**不要**在节点里
`state["node_records"].append(...)`，会重复累加。

### Cost 与 add_cost

[types.py:33](../src/babyrag/types.py#L33) 是累计成本；[types.py:92](../src/babyrag/types.py#L92) 的
`add_cost` 把一组 `NodeRecord` 求和。`finalize` 调它（[graph.py:255](../src/babyrag/graph.py#L255)），
写入 `state["cost"]`。

注意 `NodeRecord.llm_calls` 在缓存命中时是 0、`cache_hits` 是 1
（[llm.py:129](../src/babyrag/llm.py#L129)），所以总调用数不会被缓存虚高。

## Check yourself

1. 为什么 `PRICE_INPUT` 要除以 `1e6`？
2. `Settings` 是 frozen 的，那 `variant` 怎么切？
3. 为什么 `RAGState` 里的 list 字段要用 `Annotated[..., operator.add]`？

### Answers

1. 价格表按“每百万 token”报价，代码按“每 token”乘；不除会放大 100 万倍。
2. 不能改实例；要在 `load_settings()` 之前设置 `BABYRAG_VARIANT`，或在测试里传 `env={...}`。
3. LangGraph 的节点返回部分更新；没有 reducer 的 list 会被**覆盖**，有 reducer 才会 append。`node_records` / `trajectory` 需要累计，所以显式声明。
