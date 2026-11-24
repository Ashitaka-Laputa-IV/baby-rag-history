# rag-ladder 语料与评测集评估

> **实现前研究，题库与语料规模已变化**；现状以 [`../docs/results.md`](../docs/results.md) 为准。

> 评估对象：`C:\Users\Ashit\Documents\GitHub\rag-ladder`（只读，未改动任何文件）
> 评估日期：2026-11-20
> 数据口径：`chars` = Python `len(str)`；`tokens` = `cl100k_base`；文档 = 中文维基条目；切块 = 512 token / 64 overlap。
> 验证方式：全部数字由脚本实际读取 `corpus.jsonl`、`chunks.jsonl`、`queries.jsonl`、`results/runs/m1_answers.jsonl` 计算，未靠估计。✅ = 实测；🟡 = 判断/估计。

## 0. 结论先行

1. **语料本身干净可用**：200 篇、869 chunk、5 个主题各 40 篇；无重复文本/标题，license 与 URL 100% 齐全。✅
2. **查询集只有 30 题，不是任务说明和 README 宣称的 150 题**；`README.md` 与 `search/evaluation.md` 都写 150 题（40/25/30/30/25），当前文件是 30 题（12/6/6/3/3）的试点批次。✅ **这是全报告最重要的发现。**
3. **当前 30 题对 M1 已经没有区分度**：已跑的 M1 结果 29/30 正确（96.7%），27 道可答题的 gold evidence chunk 全部进入 top-5（27/27，100%）。RAPTOR / GraphRAG / CRAG / Agentic 在这套题上几乎没有可提升空间。✅
4. **512-token chunk + top-5 足以回答单跳与专名题**：27 道有 evidence 的题中，evidence 100% 完整落在至少一个 chunk 内；top-5 上下文中位数 2,236 token，远低于 4,000 token 预算。✅
5. **唯一明显失败的是全局聚合题 q025**（3 篇文档的人口比较），dense top-5 只覆盖 2/3 篇，且 5 个槽位被同一篇文档占掉多个。✅
6. **建议**：10 分钟冒烟用 **50 docs / 223 chunks / 15 题**；完整评测用现有 **200 docs / 869 chunks / 30 题** 跑通，但在对外宣称“六范式阶梯”前必须补到 150 题，并加入更难的题（去标题线索、3 跳以上、5 篇以上全局题、真正不存在的实体）。✅

---

## 1. 精确字段 schema

三个文件都是 UTF-8、无 BOM、LF 换行、每行一个 JSON 对象；所有行 key 集合一致，无 null。✅

### 1.1 `data/corpus.jsonl`（200 行）

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | str | `{theme}-{16位hex}`，由 `blake2b(seed:title)` 生成，稳定 |
| `title` | str | 维基条目标题，2–23 字 |
| `theme` | str | 枚举：`person` / `geo` / `tech` / `hist` / `art` |
| `text` | str | 清洗后正文，800–3,385 字 |
| `chars` | int | 恒等于 `len(text)`（200/200 验证） |
| `url` | str | 中文维基 URL，200/200 为 `https://zh.wikipedia.org/...` |
| `license` | str | 200/200 为 `CC-BY-SA-4.0` |

**注意**：实际字段是 `url`，不是任务说明里的 `source_url`；没有单独的 `theme?` 可选字段，`theme` 必有。`text` 内含段落换行（每篇中位 27 个 `\n`）。

样例（`text` 截断）：

```json
{"doc_id":"person-744074a4bc14901f","title":"增禄","theme":"person",
 "text":"增禄，字学亭，号鹤舲，胡氏，汉军正蓝旗人（属正蓝旗汉姓满洲旗人）。清朝官员，进士出身。\n\n生平\n笔帖式出身，道光十九年……",
 "chars":864,"url":"https://zh.wikipedia.org/wiki/%E5%A2%9E%E7%A5%BF","license":"CC-BY-SA-4.0"}
```

### 1.2 `data/chunks.jsonl`（869 行）

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | str | `{doc_id}:{seq:03d}`，如 `person-744074a4bc14901f:000` |
| `doc_id` | str | 外键到 corpus |
| `title` | str | 冗余自文档 |
| `theme` | str | 冗余自文档 |
| `seq` | int | 每篇从 0 连续递增，无空洞 |
| `text` | str | chunk 原文，等于 `doc.text[start:end]`（869/869 验证） |
| `start` / `end` | int | 字符偏移；首块 start=0，末块 end=len(text)（200/200 覆盖全文） |
| `tokens` | int | `cl100k_base` token 数，51–512 |
| `chars` | int | 恒等于 `len(text)` |

**注意**：token 字段名是 `tokens`，不是 `token_count`。没有 `token_count?` 可选字段。

样例（`text` 截断）：

```json
{"chunk_id":"person-744074a4bc14901f:000","doc_id":"person-744074a4bc14901f","title":"增禄","theme":"person",
 "seq":0,"text":"增禄，字学亭，号鹤舲，胡氏，汉军正蓝旗人……","start":0,"end":384,"tokens":490,"chars":384}
```

### 1.3 `eval/queries.jsonl`（30 行）

| 字段 | 类型 | 说明 |
|---|---|---|
| `qid` | str | `q001`–`q030` |
| `type` | str | 枚举：`single` / `entity` / `multi` / `global` / `none` |
| `theme` | str | 题目标注主题（与 gold doc 主题不必一致，见 q019） |
| `question` | str | 中文问题，13–51 字 |
| `gold_answer` | str | 参考答案；`none` 题为 `"无法回答"` |
| `gold_doc_ids` | list[str] | 1 / 2 / 3 个；`none` 题为 `[]` |
| `evidence` | str | 支持答案的原文子串；`none` 题为 `""` |
| `should_abstain` | bool | 仅 `none` 题为 `true` |

**注意**：没有 `gold_chunk_ids`，这与 `search/evaluation.md` 里“queries.jsonl 含 gold_chunk_ids”的设计不一致。当前只能靠 `doc_id` + `evidence` 子串反推 chunk。

样例：

```json
{"qid":"q001","type":"single","theme":"person",
 "question":"增禄的父亲咸成曾任什么官职？",
 "gold_answer":"福建泉州府厦门海防同知",
 "gold_doc_ids":["person-744074a4bc14901f"],
 "evidence":"父咸成（号澜峰），隶正蓝旗汉军文泰佐领下，福建泉州府厦门海防同知；",
 "should_abstain":false}
```

---

## 2. 语料统计

### 2.1 主题与长度

| theme | 文档数 | 字符中位（范围） | token 中位（范围） | chunk 数 | chunk/篇中位 |
|---|---:|---:|---:|---:|---:|
| person | 40 | 1,177（829–3,035） | 1,564（983–3,754） | 166 | 4 |
| geo | 40 | 1,416（806–3,360） | 1,687（989–4,112） | 193 | 4 |
| tech | 40 | 1,318（800–3,148） | 1,498（792–3,697） | 155 | 4 |
| hist | 40 | 1,263（850–3,164） | 1,682（1,037–3,943） | 178 | 4 |
| art | 40 | 1,522（856–3,385） | 1,823（820–4,636） | 177 | 4 |
| **合计** | **200** | **1,340（800–3,385）** | **1,650（792–4,636）** | **869** | **4** |

- 语料总计 **309,639 字符 / 375,499 tokens**，token/char ≈ **1.213**；单篇 800–3,385 字符、792–4,636 tokens（中位 1,650）。
- 文档长度人为限制在 800–4,000 字（manifest 的过滤条件），实际最大 3,385 字；因此每篇只有 2–10 个 chunk，**层级很浅**，这是 RAPTOR/GraphRAG 难以体现收益的结构性原因。🟡
- 每篇 chunk 数分布：2 块 14 篇、3 块 74 篇、4 块 38 篇、5 块 28 篇、6 块 16 篇、7 块 13 篇、8 块 14 篇、9 块 2 篇、10 块 1 篇。
- chunk 长度：51–512 tokens（中位 478），40–719 字符（中位 396）；**19 个 chunk < 100 tokens**，都是文末尾块。

### 2.2 标题样例

| theme | 标题（前 8–10 个） |
|---|---|
| person | 增禄、李崇禧、郑效峰、陈小云、胡晓媛、汪海林、曲同丰、妮可·里奇、刘健挺、黄朝天 |
| geo | 云岭、平凉县、薄扶林 (选区)、布兰太尔 (南拉纳克郡)、淘大 (选区)、鲁瓦 (索姆省)、布斯拉、维里亚、桂东县、五方礁 |
| tech | 开姆尼茨工业大学、P-8波赛顿海上巡逻机、国际和平妇幼保健院、分子中的原子理论、土耳其国铁E43000型电力机车、2028年7月22日日食、Windows 8开发历史、ISO/IEC 27002、生物组织光学窗口、039型潜艇 |
| hist | 扎巴依喀勒山空难、安成郡、弗拉基米尔·列宁全联盟先锋组织、尼泊尔王室枪击事件、重俊之变、天津蓟县莱德商厦火灾、欧洲中世纪狩猎、南海1018轮事件、环加拿大航空831号班机空难、丹纳航空992号班机空难 |
| art | 卡加利牛仔节、桃花岛 (金庸小说)、SILVER LINK.、和敬公主府、野蛮游戏、金曲音乐节、情约一天、中国体育博物馆、苏珊·凯文、机·丹尼尔·奥利瓦 |

### 2.3 许可、URL、重复、标题泄漏

| 检查项 | 结果 |
|---|---|
| license 覆盖 | 200/200 = `CC-BY-SA-4.0` |
| URL 覆盖 | 200/200；全部 `https://zh.wikipedia.org` |
| 空文本 | 0 |
| 完全重复文本 | 0 组（SHA-256） |
| 重复标题 | 0 组 |
| 相同 100 字前缀 | 0 组 |
| `title` 出现在正文 | **161/200（80.5%）** |
| 正文以标题开头 | 134/200（67.0%） |

`title` 出现在正文意味着向量/BM25 都能在“题干含标题”时获得近乎免费的强信号；这也是下面查询集 retrieval 偏高的原因之一。

切块质量：200/200 篇全文被 chunk 无缝覆盖，无 seq 空洞，`chunk.text == doc.text[start:end]` 869/869 成立。语料已清理参考文献/外部链接段（0 篇含“参考文献”），无 `。。` 等清洗残渣。

---

## 3. 查询集统计

### 3.1 总量与分布

**实际 30 题，不是 150 题。** 行数、最大 `qid`（q030）、`results/runs/m1_answers.jsonl` 行数三者一致，都是 30。✅

| type | 题数 | person | geo | tech | hist | art |
|---|---:|---:|---:|---:|---:|---:|
| single | 12 | 2 | 3 | 3 | 2 | 2 |
| entity | 6 | 2 | 1 | 1 | 1 | 1 |
| multi | 6 | 0 | 1 | 1 | 2 | 2 |
| global | 3 | 0 | 1 | 0 | 1 | 1 |
| none | 3 | 1 | 0 | 1 | 0 | 1 |
| **合计** | **30** | **5** | **6** | **6** | **6** | **7** |

- `should_abstain`：`false` 27、`true` 3（全部是 `none` 题）。
- 问题长度：13–51 字，中位 21，均值 25.7；`≤20` 14 题、`21–35` 9 题、`36–50` 6 题、`>50` 1 题。
- 30 个问题全部唯一；无空问题；无空 `gold_answer`。
- `evidence` 长度：11–76 字，中位 23 字；3 条为空（`none` 题）。

### 3.2 gold 文档

| `gold_doc_ids` 数量 | 题数 | 说明 |
|---:|---:|---|
| 0 | 3 | q028–q030（`none`） |
| 1 | 18 | 12 single + 6 entity |
| 2 | 6 | 全部 multi |
| 3 | 3 | 全部 global |

- **所有 gold doc_id 都存在于 corpus**（0 个脏 ID）。✅
- 所有 gold 文档的 `doc_id` 前缀与其自身 `theme` 一致。需要注意的是，**q019 的 query theme 是 `hist`，其 gold 却含一篇 `person` 文档**（叶举）和一篇 `hist` 文档（两广护国军都司令部）；按 theme 分层统计时要把 query theme 与文档主题分开看。✅
- 多 gold 题均为“2 文档比较”或“3 文档聚合”，不是同一事实的多篇重复来源。

### 3.3 gold chunk 能否反推

当前没有 `gold_chunk_ids`，只能用 `evidence in chunk.text` 反推。✅

| 情况 | 题数 | 明细 |
|---|---:|---|
| evidence 恰好落在 1 个 chunk | 25 | 可直接唯一确定 gold chunk |
| evidence 落在 2 个 chunk（64-token overlap） | 2 | q003、q019；两个 chunk 都含同一句 |
| evidence 在文档中但跨 chunk 断裂 | 0 | 未发现 |
| 无 evidence（3 道 `none` 题） | 3 | q028–q030，`evidence=""`，没有可解析的 gold chunk |

结论：**27 道有 evidence 的题都能从 `doc_id` + evidence 子串解析出至少一个 gold chunk**；但 q003/q019 存在两个候选，做 chunk 级 `evidence_recall` 时必须先约定“取 seq 最小”或“两个都算命中”，否则会把评测口径的不确定性误当成模型错误。🟡

### 3.4 答案与 evidence 的关系

| 检查项 | 结果 |
|---|---|
| `gold_answer` 是 `evidence` 的子串 | 18/27 可答题（66.7%） |
| `gold_answer` 是任一 gold 文档正文的子串 | 25/27 可答题（92.6%） |
| 需要跨文档计算/归一化后才有答案 | 2 题（q020、q025） |
| 答案为 `"无法回答"` | 3 题（q028–q030，不计入上两行） |

`gold_answer` 不在 gold 文档里的 5 题 = 3 道拒答题 + q020（面积差 14.30 = 15.23 − 0.93）+ q025（人口差 10,293 = 13,791 − 3,498）。这两道计算题必须用数值容差或 LLM judge，不能做字符串匹配。

### 3.5 标题线索泄漏（重要）

- **26/30（86.7%）的问题包含一个 corpus 中完全一致的标题**；按题型：single 9/12、entity 6/6、multi 5/6、global 3/3、none 3/3。
- 27 道可答题中，**23 道的问题包含其 gold 标题**；只有 q005、q007、q008、q021 不含完整 gold 标题。
- 结合 80.5% 文档正文包含标题，当前检索分数被明显抬高，M1 的 100% evidence recall 不能代表真实难度。✅

---

## 4. 题型 × RAG 能力映射

> 下表的“M1 实测”来自 `results/runs/m1_answers.jsonl`：人工逐题核对答案，q009 的“17人”与 gold“17名”按语义正确计。✅

| type | 题数 | 目标能力 / 对应 rung | 当前数据是否支撑 | M1 实测 |
|---|---:|---|---|---:|
| `single` | 12 | 单跳 dense（M1 基础） | 12/12 evidence chunk 在 top-5；每题 1 篇文档 | 12/12 |
| `entity` | 6 | 专名/数字，BM25 或 hybrid（M2） | 6/6 dense 命中；BM25 字符 bigram R@1 为 5/6、R@5 6/6 | 6/6 |
| `multi` | 6 | 2 跳、跨文档组合（M6 / M4） | 6/6 的两篇 gold 文档都进了 top-5；dense 已能答 | 6/6 |
| `global` | 3 | 全局聚合、跨 3 篇比较（M3 RAPTOR / M4 摘要） | 2/3 能在 top-5 覆盖全部 3 篇；q025 只覆盖 2/3 | 2/3 |
| `none` | 3 | 无答案拒答（M5 CRAG） | 3/3 的干扰文档存在且确实无答案关键信息 | 3/3 正确拒答 |
| **合计** | **30** | — | — | **29/30（96.7%）** |

**关键判断**：能力标签是“设计意图”，不是“实测区分度”。由于 dense + 普通 prompt 已经解决了 12/12 single、6/6 entity、6/6 multi、3/3 none，只有 global 的 1 道题体现差距。M1 与 M6 的差距上限只有 3.3 个百分点，无法支撑“逐级量出收益”的主榜叙事。🟡

按设计意图归因：dense 覆盖 18 题（12 single + 6 entity），BM25/hybrid 加强其中 6 道 entity，agentic 面向 6 道 multi，RAPTOR/global 面向 3 道 global，CRAG 面向 3 道 none。但实测 M1 已答对 29/30（包括全部 6 道 multi），后五级实际只能争 q025 这 1 题（3.3 个百分点），同时还有把已答对的题做坏的风险。

对照 `search/evaluation.md` 的目标构成（150 题）：

| type | 目标题数 | 现有 | 缺口 |
|---|---:|---:|---:|
| single | 40 | 12 | 28 |
| entity | 25 | 6 | 19 |
| multi | 30 | 6 | 24 |
| global | 30 | 3 | 27 |
| none | 25 | 3 | 22 |
| **合计** | **150** | **30** | **120** |

---

## 5. 200 篇 demo 的可行性

### 5.1 单跳题：512-token chunk + top-5 够不够？

**够，而且余量很大。** ✅

- evidence 中位长度只有 23 字，最长 76 字；27/27 都能完整落在某个 chunk 内（其中 q003、q019 因 64-token overlap 同时落在两个 chunk），0 条被 chunk 边界切断。
- chunk 上限 512 token，top-5 理论上限 2,560 token；实测 dense top-5 上下文 **中位 2,236、最大 2,475 tokens**，低于 `MAX_CONTEXT_TOKENS=4000`。
- 现有 M1 run（bge-m3 + Chroma）的检索结果：从其 `retrieved` 字段复算，**27/27 道可答题的 evidence chunk 都在 top-5**（100%）。
- 纯字符 bigram BM25 基线：按 27 道可答题计，chunk Recall@1 = 19/27（70%），Recall@5 = 26/27（96%）；doc Recall@5 = 27/27（100%）。若把 3 道 `none` 题也算进分母，则是 19/30、26/30、27/30。这说明中文短文档上词法检索已经不弱，M2 的 BM25 分支很难拉开差距。🟡

因此，单跳题的检索上限是 12/12，生成实测也是 12/12；**single + entity 的 baseline ceiling ≈ 100%**。

### 5.2 整体 baseline ceiling

以“gold evidence 进入 top-5 且答案可被读出”为上限：

| 能力 | 上限 | 依据 |
|---|---:|---|
| single | 12/12 | evidence 全部命中，答案均在 chunk 中 |
| entity | 6/6 | evidence 全部命中；答案均可在 chunk 中读出 |
| multi | 6/6 | 两篇 gold 文档均在 top-5；跨文档比较可完成 |
| global | 2/3 | q025 只覆盖 3 篇中的 2 篇，且缺少拉巴塞人口 |
| none | 3/3 | 3 篇干扰文档均无答案信息 |
| **总体** | **29/30 ≈ 96.7%** | 与已跑 M1 结果一致 |

**结论**：这不是“能不能跑”的问题，而是“跑了也看不出阶梯”的问题。30 题下 1 题 = 3.3 个百分点；96.7% 的 Wilson 95% 置信区间约为 **83%–99%（±8 个百分点）**，六个范式之间的差距会完全被噪声吞掉。🟡

### 5.3 gold 文档/答案有歧义的题

| qid | 问题 | 歧义点 | 影响 |
|---|---|---|---|
| q019 | 1916 年叶举在哪个机构任参谋长 | `theme=hist`，但 gold 是 1 篇 person + 1 篇 hist；跨主题 | theme 分层指标会把 person 文档算进 hist |
| q005 | 司马文击败民建联哪位候选人 | 题干不含标题“薄扶林 (选区)”，检索需靠语义 | 无标题线索；当前 dense 仍命中 |
| q020 | 弗雷旺面积比埃丹大多少 | evidence 只覆盖弗雷旺（15.23），埃丹的 0.93 在另一篇；答案是算术 | evidence_recall 只测到一半事实 |
| q025 | 三市镇 2018 人口差 | evidence 只覆盖贝尔克（13,791）；答案需三篇人口并做减法；dense top-5 中 4 个槽位给了弗雷旺、1 个给贝尔克，拉巴塞 0 | 当前唯一失败题；也是 global 能力的关键样本 |
| q027 | 三部影视哪部先以电视剧首播 | 3 篇 gold；global 题的 5 个槽位要装下 3 篇文档，容易漏一篇 | 检索槽位竞争，聚合所需事实可能不全 |
| q028–q030 | 李崇禧/开姆尼茨/喷嚏大魔王 的缺失属性 | gold_doc_ids 为空；必须靠“文档存在但属性不存在”拒答 | 不能区分“不知道”与“检索没找到” |
| q009 | 空难遇难人数 | gold “17名” vs 实际答案 “17人” | 字符串匹配会误判，需归一化 |
| q011/q013/q014 | 年份题 | “2008年/1955年/1985年”分别出现在 24/12/12 篇文档中 | 答案字符串在语料中不唯一，必须依赖 gold doc 上下文 |

另一个结构性歧义：dense top-5 平均只有 **2.13 篇不同文档**（5 个槽位被同篇文档的多个 chunk 占满）。对 3 篇文档的 global 题，这是比“模型不会聚合”更直接的失败原因。✅

---

## 6. 子集建议

### 6.1 （a）10 分钟冒烟测试

目标：验证数据管线、embedding、Chroma、LLM 调用、引用与指标脚本能否端到端跑通，不追求区分度。

| 项目 | 数值 |
|---|---|
| 文档 | **50 篇**（5 主题 × 10 篇） |
| chunk | **223 个** |
| 语料规模 | 81,645 字符 / 98,209 tokens |
| 问题 | **15 题**：single 4、entity 3、multi 3、global 2、none 3 |
| 预计耗时 | 索引 1–2 分钟 + 15 次生成 1–2 分钟，含调试 < 10 分钟 |
| 预计成本 | < $0.01（按现有 M1 单价） |

题目（可直接用 `qid` 过滤）：

```
single: q001 q004 q007 q011
entity: q013 q016 q018
multi:  q019 q021 q024
global: q025 q026
none:   q028 q029 q030
```

选文档规则（确定性，便于复现）：

1. 取上述 15 题的 `gold_doc_ids` 并集（22 篇）；
2. 对 `none` 题，加入题干所点名的文档（q028 李崇禧、q029 开姆尼茨工业大学、q030 喷嚏大魔王）；
3. 每个主题按 `doc_id` 升序补到 10 篇。

结果正好 50 篇，且 q019 的跨主题 person 文档、q025 的三篇法国市镇、q021 的两款步枪都在内。文件建议：

```
data/smoke/corpus.jsonl      # 50 行
data/smoke/chunks.jsonl      # 223 行
eval/queries-smoke.jsonl     # 15 行
```

### 6.2 （b）完整 $10 预算评测

| 方案 | 文档 | chunk | 问题 | 说明 |
|---|---:|---:|---:|---|
| 现状全量 | 200 | 869 | 30 | **先跑这个**，用于验证全链路与复现 M1 的 29/30 |
| 目标主榜 | 200 | 869 | 150 | 需先补 120 题；`$10` 预算远远够 |

成本实测：已跑的 30 题 M1 消耗 41,891 input + 4,781 output tokens，按仓库定价 **$0.0092**。线性外推 150 题约 **$0.05**；即使 6 个范式、agentic 3–5 倍调用、再加 judge，仍在 **$1–2** 量级，与 `search/evaluation.md` 的 $1.8 计划一致。🟡 一次性索引成本也可忽略：全库 375,499 tokens，按常见 bge-m3 单价远低于 $0.1。🟡

建议文件：

```
data/corpus.jsonl             # 200 行，完整语料（全量评测用）
data/chunks.jsonl             # 869 行
eval/queries.jsonl            # 30 行，现状
```

如果坚持“完整评测必须是 150 题”，则当前数据只能算 **pilot**；对外结果必须标注 `n=30`，不能与 150 题主榜混用。

---

## 7. 数据准备计划（baby-rag-history）

### 7.1 复制清单

```bat
cd C:\Users\Ashit\Documents\GitHub\baby-rag-history
mkdir data
mkdir eval

copy C:\Users\Ashit\Documents\GitHub\rag-ladder\data\corpus.jsonl        data\corpus.jsonl
copy C:\Users\Ashit\Documents\GitHub\rag-ladder\data\chunks.jsonl        data\chunks.jsonl
copy C:\Users\Ashit\Documents\GitHub\rag-ladder\data\manifest.json       data\manifest.json
copy C:\Users\Ashit\Documents\GitHub\rag-ladder\data\manifest-chunks.json data\manifest-chunks.json
copy C:\Users\Ashit\Documents\GitHub\rag-ladder\eval\queries.jsonl       eval\queries.jsonl
```

校验 SHA-256（与 manifest 对齐，防止复制损坏）：

| 文件 | SHA-256 |
|---|---|
| `corpus.jsonl` | `f93ed72df40c1831ddbcaeda47a66d3874182f4364bfda4362cd731e05c07094` |
| `chunks.jsonl` | `173c410c19f66d7c616b6a70f560336cea3bf2f19c7e503832071a3689a3c2b5` |
| `queries.jsonl` | `602fbe9574ec0e357fbbd8ff6f04724da6f0fa2b1652f4ebc0e12b76625d240a` |

`corpus.jsonl` 的哈希与 `manifest-chunks.json` 中的 `corpus_sha256` 一致。✅

### 7.2 冒烟子集生成片段

```python
import json
from pathlib import Path

SRC = Path(r"C:\Users\Ashit\Documents\GitHub\rag-ladder")
OUT = Path("data/smoke")
OUT.mkdir(parents=True, exist_ok=True)

corpus = [json.loads(l) for l in (SRC / "data/corpus.jsonl").open(encoding="utf-8")]
chunks = [json.loads(l) for l in (SRC / "data/chunks.jsonl").open(encoding="utf-8")]
queries = [json.loads(l) for l in (SRC / "eval/queries.jsonl").open(encoding="utf-8")]

QIDS = {"q001","q004","q007","q011","q013","q016","q018",
        "q019","q021","q024","q025","q026","q028","q029","q030"}
qs = [q for q in queries if q["qid"] in QIDS]

need = {d for q in qs for d in q["gold_doc_ids"]}
need |= {d["doc_id"] for d in corpus
         for q in qs if not q["gold_doc_ids"] and d["title"] in q["question"]}

selected = set(need)
for theme in ["person", "geo", "tech", "hist", "art"]:
    have = [d for d in corpus if d["theme"] == theme and d["doc_id"] in selected]
    for d in sorted((d for d in corpus
                     if d["theme"] == theme and d["doc_id"] not in selected),
                    key=lambda x: x["doc_id"]):
        if len(have) >= 10:
            break
        selected.add(d["doc_id"])
        have.append(d)

def dump(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump(OUT / "corpus.jsonl", [d for d in corpus if d["doc_id"] in selected])
dump(OUT / "chunks.jsonl", [c for c in chunks if c["doc_id"] in selected])
dump(Path("eval/queries-smoke.jsonl"), qs)
```

输出应为 **50 篇 / 223 chunk / 15 题**（`chars` 81,645，`tokens` 98,209）。

### 7.3 `.gitignore`

现有 `.gitignore` 已包含 `.env`、`.venv`、`__pycache__`。建议追加：

```gitignore
# local corpus artifacts (regenerable / large)
data/raw/
data/chroma/
data/**/chroma/
*.parquet

# smoke subset is regenerated by script
data/smoke/

# optional: local eval runs
results/runs/
```

**保留在 Git 里**：`data/corpus.jsonl`、`data/chunks.jsonl`、`data/manifest*.json`、`eval/queries*.jsonl`。合计约 1.9 MB，适合普通 Git；不需要 LFS。

### 7.4 CC-BY-SA 4.0 归属文本

新建 `data/ATTRIBUTION.md`（或放 `NOTICE`）：

```markdown
# 数据来源与许可

本仓库的 `data/corpus.jsonl` 与 `data/chunks.jsonl` 派生自中文维基百科：

- 原始数据集：`wikimedia/wikipedia`，快照文件 `zhwiki-00002.parquet`
- 构建时间：2026-11-18
- 许可：Creative Commons Attribution-ShareAlike 4.0 International
  （CC BY-SA 4.0），https://creativecommons.org/licenses/by-sa/4.0/
- 原文作者：维基百科贡献者；每篇条目的来源链接与许可见
  `corpus.jsonl` 的 `url` 与 `license` 字段。

本仓库对原文做了以下修改：
1. 转换为简体中文（zhconv，zh-cn）；
2. 清理参考文献、分类尾注、模板残片与多余空行；
3. 过滤为 800–4000 字、按 512 token / 64 token overlap 切块。

上述派生数据同样以 CC BY-SA 4.0 发布。本仓库的源代码使用 MIT 许可，
MIT 不适用于 `data/` 下的维基百科派生文本。
```

每个文档的 `url` + `license` 已能提供逐条归属；若对外分发，再附上“本文档包含来自中文维基百科的内容，CC BY-SA 4.0”即可。修改与再分发必须继续 ShareAlike，不能把语料改成 MIT。

---

## 8. 会破坏评测的数据质量问题

按严重度排序。P0 会直接让评测结论失效，P1 会让指标不可比，P2/P3 影响解释力。

| # | 级别 | 问题 | 证据 | 建议 |
|---|---|---|---|---|
| 1 | **P0** | 查询集只有 30 题，README/评测方案宣称 150 题 | `queries.jsonl` 30 行；`README.md` 第 2 行写 150；目标构成 40/25/30/30/25 | 补 120 题，或全项目统一改口径为 30 题试点；不得把 30 题结果标成 150 题主榜 |
| 2 | **P0** | 难度天花板效应，M1 已 29/30 | evidence recall@5 = 27/27；single/entity/multi 全对；仅 q025 失败 | 增加去标题线索、3+ 跳、5+ 文档聚合、真正未知实体题；否则 M2–M6 无提升空间 |
| 3 | **P1** | 没有 `gold_chunk_ids`，无法稳定算 chunk 级 evidence_recall | 字段只有 evidence；q003/q019 的 evidence 同时命中 2 个 chunk（overlap） | 生成时写入 `gold_chunk_ids`；或规定“evidence 命中的所有 chunk 均算命中” |
| 4 | **P1** | multi/global 的 evidence 只覆盖部分 hop | q020 只给弗雷旺面积，缺埃丹；q025 只给贝尔克人口，缺拉巴塞/弗雷旺 | 每题给出 `evidence_by_doc` 或每跳一条 evidence，否则 evidence_recall 系统性偏低 |
| 5 | **P1** | 答案格式不适合字符串匹配 | 9 道数字/年份题；q008 `120 km/h` 含空格；q018/q025 含千分位逗号；q020/q025 是算术结果；q002 gold 全称 vs evidence 简称“南昌中院”；q009 `17名` vs 生成 `17人` | 用归一化（去空格/逗号/单位）+ 数值容差，或统一 LLM judge；不要用 exact match 直接下结论 |
| 6 | **P1** | 标题泄漏，检索分数虚高 | 26/30 问题含完整 corpus 标题；161/200 文档正文含标题；23/27 可答题含 gold 标题 | 增加“标题不出现在问题中”的变体；报告 retrieval 时区分 title-cued / title-masked |
| 7 | **P2** | `none` 题只有 3 道，且都是“实体存在、属性不存在” | q028–q030 都点名一个已收录文档；无“实体完全不在语料”的题 | 补 10+ 道：实体不存在、问题超出语料时间范围、需要多跳但中间实体缺失；分别测 CRAG 的不同失败模式 |
| 8 | **P2** | 主题标注与 gold 文档不一致 | q019 `theme=hist` 但含 `person` 文档；`art` 下有 苏珊·凯文 这类人物条目 | theme 只作弱分层；指标按 gold 文档主题或双标签统计 |
| 9 | **P2** | 文档过短、层级过浅，RAPTOR/GraphRAG 难有收益 | 每篇 2–10 chunk，中位 4；375k tokens 全库 | 若要论证 RAPTOR/GraphRAG，补 20+ 篇 8k–20k 字长文档，或构造跨 5–10 篇的 global 题 |
| 10 | **P2** | 全局题槽位竞争 | dense top-5 平均只有 2.13 篇不同文档；q025 只覆盖 2/3 篇 | 全局题评测时用 doc-level 去重检索、扩大 k，或明确允许 agentic 多轮检索 |
| 11 | **P3** | 上游语料模板残缺 | 11 篇 geo 文档首句为“时人口数量为人”（源 parquet 即缺数字，如贝尔克）；正文后文另有完整人口句 | 出题避开首句模板；数据清洗时不要把“数字缺失”误判为模型错误 |
| 12 | **P3** | 相似模板文档造成近重复 | 多篇法国市镇条目结构几乎相同；贝尔克/弗雷旺/拉巴塞/埃丹同属一类 | 检索评测需按 doc_id 判分，不要用答案字符串判“命中哪篇” |
| 13 | **P3** | 答案字符串在语料中不唯一 | “2008年”出现在 24 篇、“1955年”12 篇、“1985年”12 篇 | 必须结合 gold doc / chunk 判定；单看答案字符串会高估 |
| 14 | **P3** | 无题目来源、难度、标注者、split 字段 | queries schema 只有 8 个字段 | 增加 `difficulty`、`source`、`split`（dev/test）、`notes`；30 题也要固定 holdout，避免调 prompt 时过拟合 |
| 15 | **P3** | `qid` 只有 3 位数字，未来扩到 150 需保持稳定 | q001–q030 | 扩题时不要重编号已有题；新题从 q031 起 |

---

## 附：本次验证脚本

临时脚本位于 `C:\Users\Ashit\AppData\Local\Temp\opencode\`（未写入 rag-ladder）：

- `analyze1.py`：schema、语料/查询统计、evidence → chunk 解析
- `retrieval.py`：字符 bigram BM25 基线、Recall@k、上下文 token
- `deep.py`：gold doc 覆盖、evidence 泄漏、重复 chunk
- `actual.py` / `cov.py`：读取 `results/runs/m1_answers.jsonl`，算实际 dense evidence recall 与逐题答案
- `subset.py` / `smoke2.py`：冒烟子集规模与确定性选样
- `theme.py` / `misc.py` / `artifact.py`：分主题表、标题泄漏、上游模板残片

所有读取均为只读；`rag-ladder` 目录未被修改。
