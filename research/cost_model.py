# -*- coding: utf-8 -*-
import json, collections, statistics, math, random
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
def T(s): return len(enc.encode(s))

ROOT = r"C:\Users\Ashit\Documents\GitHub\rag-ladder"
rows = [json.loads(l) for l in open(ROOT + r"\eval\queries.jsonl", encoding="utf-8") if l.strip()]
chunks = [json.loads(l) for l in open(ROOT + r"\data\chunks.jsonl", encoding="utf-8") if l.strip()]

types = collections.Counter(r["type"] for r in rows)
print("ACTUAL N =", len(rows), "TYPES =", dict(types))

# scaled to 150 using measured proportions
N_TARGET = 150
scale = N_TARGET / len(rows)
proportions = {k: v / len(rows) for k, v in types.items()}
counts150 = {k: round(v * scale) for k, v in types.items()}
print("PROPORTIONS =", {k: round(v,4) for k,v in proportions.items()})
print("COUNTS150 =", counts150, "sum =", sum(counts150.values()))

# per-type question / answer token stats
per_type = {}
for t in types:
    sub = [r for r in rows if r["type"] == t]
    qt = statistics.mean(T(r["question"]) for r in sub)
    per_type[t] = {"n30": len(sub), "n150": counts150[t], "q_tokens": qt}
    print(t, len(sub), round(qt,1))

# ---- context measurement: 100 random draws of 5 chunks, build_context format ----
random.seed(42)
def ctx_tokens(n=5):
    hits = random.sample(chunks, n)
    pieces = [f"[{i+1}] （{h['title']}）{h['text']}" for i, h in enumerate(hits)]
    return T("\n\n".join(pieces))

draws = [ctx_tokens(5) for _ in range(300)]
CTX5 = statistics.mean(draws)
print("CTX5 mean", round(CTX5,1), "median", statistics.median(draws), "min", min(draws), "max", max(draws))

# ---- prompt templates, measured exactly ----
BASE_SYS = open(ROOT + r"\src\ragladder\prompts\m1.md", encoding="utf-8").read().split("\n---\n")[0].strip()
BASE_USER = "问题：{question}\n\n资料：\n{context}"
print("BASE_SYS tokens", T(BASE_SYS), "BASE_USER skeleton", T(BASE_USER))

HYDE_SYS = "你是维基百科编辑。根据问题写一段可能出现在百科条目中的资料性文字，只陈述事实，不要回答问题本身，不要编号。"
HYDE_USER = "问题：{question}\n\n请写一段 200-350 字的资料性文字。"
MQ_SYS = "你是检索查询改写器。把用户问题改写成多个互补的中文检索查询，覆盖同义词、专名全称和隐含子问题。只输出 JSON 数组，不要解释。"
MQ_USER = "问题：{question}\n\n生成 3 条不同的检索查询。"
CTX_SYS = "你是文档编辑。下面是整篇文档和其中一个片段。请写出 50-100 字说明，指出该片段在全文中讲的是什么、包含哪些关键实体或数字，以便改善检索。只输出说明文字。"
CTX_USER = "文档标题：{title}\n\n整篇文档：\n{doc}\n\n片段：\n{chunk}"
RAPTOR_SYS = "你是摘要器。把给定片段合并成一段 150-250 字的中文摘要，保留所有人名、机构名、地名、时间、数字和因果链，不要添加原文没有的信息。"
RAPTOR_USER = "片段：\n{group}"
GRADE_SYS = "你是检索质量评估器。判断给定资料能否回答用户问题。只输出 correct、ambiguous 或 incorrect 三个词之一。"
GRADE_USER = "问题：{question}\n\n资料：\n{context}"
REWRITE_SYS = "你是查询改写器。上一轮检索没有找到可用资料。请换一种说法、补充同义词或拆成更具体的子问题，只输出一条新查询。"
REWRITE_USER = "原问题：{question}\n上一轮查询：{old_query}"
ROUTER_SYS = "你是路由器。根据问题复杂度选择处理方式，只输出一个标签：simple（单跳事实）、hybrid（专名或编号）、selfcheck（可能检索不到）、agent（多跳）。"
ROUTER_USER = "问题：{question}"
JUDGE_SYS = "你是严格的中文问答评分员。比较回答 A 与回答 B 与参考答案的一致性、事实准确性和引用支持度。只输出 JSON：{\"winner\":\"A\"|\"B\"|\"tie\",\"reason\":\"一句话\"}。"
JUDGE_USER = "问题：{question}\n参考答案：{gold}\n\n回答 A：\n{a}\n\n回答 B：\n{b}"

for name, s, u in [("hyde", HYDE_SYS, HYDE_USER), ("mq", MQ_SYS, MQ_USER),
                   ("ctx", CTX_SYS, CTX_USER), ("raptor", RAPTOR_SYS, RAPTOR_USER),
                   ("grade", GRADE_SYS, GRADE_USER), ("rewrite", REWRITE_SYS, REWRITE_USER),
                   ("router", ROUTER_SYS, ROUTER_USER), ("judge", JUDGE_SYS, JUDGE_USER)]:
    print(f"PROMPT {name}: sys={T(s)} user_skeleton={T(u)} total_skeleton={T(s)+T(u)}")

# avg doc tokens (group chunks by doc; approximate doc = union, use sum tokens minus overlap)
docs = collections.defaultdict(list)
for c in chunks: docs[c["doc_id"]].append(c)
doc_tokens = []
for d, cs in docs.items():
    cs.sort(key=lambda x: x["seq"])
    # sum tokens minus overlap (64 per boundary, approximate as manifest says median overlap 46)
    doc_tokens.append(sum(c["tokens"] for c in cs) - 46 * (len(cs) - 1))
print("DOC tokens mean", round(statistics.mean(doc_tokens),1), "median", statistics.median(doc_tokens), "max", max(doc_tokens))
print("CHUNK tokens mean", round(sum(c["tokens"] for c in chunks)/len(chunks),1))
print("TOTAL chunk tokens", sum(c["tokens"] for c in chunks))

# question token total avg for 150 scaled
avg_q = sum(per_type[t]["q_tokens"] * counts150[t] for t in types) / N_TARGET
print("AVG_Q_150", round(avg_q,2))

# ---- model ----
PRICE = {"in": 0.15/1e6, "out": 0.60/1e6, "cache": 0.003/1e6}
def cost(i, o, cached=0):
    return (i - cached) * PRICE["in"] + cached * PRICE["cache"] + o * PRICE["out"]
def usd(x): return round(x, 6)

# answer output tokens by type (assumption, labelled)
ANS = {"single": 120, "entity": 45, "multi": 170, "global": 420, "none": 12}
avg_out = sum(ANS[t] * counts150[t] for t in types) / N_TARGET
print("AVG_OUT_150", round(avg_out,1))

sysb = T(BASE_SYS); ub = T(BASE_USER) - T("{question}") - T("{context}")  # fixed part
# user skeleton fixed text: "问题：" + "\n\n资料：\n"
print("base user fixed text tokens", T(BASE_USER.format(question="", context="")))
BASE_FIXED = T(BASE_USER.format(question="", context=""))
BASE_IN = sysb + BASE_FIXED + avg_q + CTX5 + 8  # +8 chat-format overhead
print("BASE_IN tokens", round(BASE_IN,1))

# sensitivity: planned composition 40/25/30/30/25 from docs/evaluation.md
PLANNED = {"single": 40, "entity": 25, "multi": 30, "global": 30, "none": 25}
avg_q_plan = sum(per_type[t]["q_tokens"] * PLANNED[t] for t in PLANNED) / 150
avg_out_plan = sum(ANS[t] * PLANNED[t] for t in PLANNED) / 150
print("PLANNED avg_q", round(avg_q_plan,1), "avg_out", round(avg_out_plan,1))
print("PLANNED base_in", round(sysb + BASE_FIXED + avg_q_plan + CTX5 + 8, 1))

hyde_in = T(HYDE_SYS) + T(HYDE_USER.format(question="")) + avg_q
mq_in = T(MQ_SYS) + T(MQ_USER.format(question="")) + avg_q
grade_in = T(GRADE_SYS) + T(GRADE_USER.format(question="", context="")) + avg_q + CTX5
rewrite_in = T(REWRITE_SYS) + T(REWRITE_USER.format(question="", old_query="")) + avg_q + avg_q
router_in = T(ROUTER_SYS) + T(ROUTER_USER.format(question="")) + avg_q
raptor_in_l1 = T(RAPTOR_SYS) + T(RAPTOR_USER.format(group="")) + 9 * 432.1
raptor_in_l2 = T(RAPTOR_SYS) + T(RAPTOR_USER.format(group="")) + 9 * 220
DOC_MEAN = statistics.mean(doc_tokens)
ctx_in = T(CTX_SYS) + T(CTX_USER.format(title="", doc="", chunk="")) + 8 + DOC_MEAN + 432.1
ctx_doc_part = DOC_MEAN
print("CTX_INDEX in", round(ctx_in,1), "doc part", round(DOC_MEAN,1))

R = {}
R["constants"] = dict(ctx5=CTX5, base_in=BASE_IN, avg_q=avg_q, avg_out=avg_out,
                      hyde_in=hyde_in, mq_in=mq_in, grade_in=grade_in, rewrite_in=rewrite_in,
                      router_in=router_in, raptor_in_l1=raptor_in_l1, raptor_in_l2=raptor_in_l2,
                      ctx_in=ctx_in, ctx_doc_part=ctx_doc_part,
                      prompt_tokens=dict(base_sys=sysb, base_user_fixed=BASE_FIXED,
                                         hyde=T(HYDE_SYS)+T(HYDE_USER.format(question="")),
                                         mq=T(MQ_SYS)+T(MQ_USER.format(question="")),
                                         grade=T(GRADE_SYS)+T(GRADE_USER.format(question="",context="")),
                                         rewrite=T(REWRITE_SYS)+T(REWRITE_USER.format(question="",old_query="")),
                                         router=T(ROUTER_SYS)+T(ROUTER_USER.format(question="")),
                                         judge=T(JUDGE_SYS)+T(JUDGE_USER.format(question="",gold="",a="",b=""))))

# per stage: index (calls, in, out, cached_read) and query (calls, in, out)
st = {}
# S1
st["S1"] = dict(name="Naive dense RAG", idx=(0,0,0,0), q=(1, BASE_IN, avg_out))
# S2
st["S2"] = dict(name="Hybrid BM25+dense+RRF+rerank", idx=(0,0,0,0), q=(1, BASE_IN, avg_out))
# S3 HyDE
st["S3"] = dict(name="HyDE", idx=(0,0,0,0), q=(2, BASE_IN + hyde_in, avg_out + 300))
# S4 multi-query: rewrite call + gen. context slightly larger? same
st["S4"] = dict(name="Multi-query / RAG-Fusion (3 rewrites)", idx=(0,0,0,0), q=(2, BASE_IN + mq_in, avg_out + 80))
# S5 contextual: index 869 calls, doc cached after first per doc
# cached tokens: 669 calls read doc part (1877), 200 pay full
s5_calls = 869
s5_in_total = s5_calls * ctx_in
s5_cached = (869 - 200) * ctx_doc_part  # 669 * 1877
st["S5"] = dict(name="Contextual Retrieval", idx=(s5_calls, s5_in_total, s5_calls*75, s5_cached),
                q=(1, BASE_IN + 5*75, avg_out))  # prefix included in context
# S6 RAPTOR: 869/9 = 96.6 -> 97 L1, 97/9=10.8 -> 11 L2
l1, l2 = 97, 11
s6_calls = l1 + l2
s6_in = l1 * raptor_in_l1 + l2 * raptor_in_l2
s6_out = s6_calls * 250
st["S6"] = dict(name="RAPTOR summary tree (2 levels)", idx=(s6_calls, s6_in, s6_out, 0),
                q=(1, BASE_IN + 250, avg_out))
# S7 CRAG / Self-RAG: grade + gen + 30% (rewrite + regen + regrade)
f = 0.30
s7_calls = 2 + f*3
s7_in = grade_in + BASE_IN + f*(rewrite_in + BASE_IN + grade_in)
s7_out = 15 + avg_out + f*(30 + avg_out + 15)
st["S7"] = dict(name="Self-RAG/CRAG grading+rewrite+abstain", idx=(0,0,0,0), q=(s7_calls, s7_in, s7_out))
# S8 agentic: 3.2 calls, 3x input
st["S8"] = dict(name="Agentic ReAct (max 5, avg 3.2)", idx=(0,0,0,0), q=(3.2, 3.0*BASE_IN, 3*60 + avg_out))
# S9 router: router + weighted routed pipeline
w = {"S1":0.40, "S2":0.30, "S7":0.20, "S8":0.10}
route_in = sum(w[k]*st[k]["q"][1] for k in w)
route_out = sum(w[k]*st[k]["q"][2] for k in w)
route_calls = sum(w[k]*st[k]["q"][0] for k in w)
st["S9"] = dict(name="Adaptive router", idx=(0,0,0,0), q=(1 + route_calls, router_in + route_in, 8 + route_out))

# GraphRAG separate estimate
graph_idx_calls = 869 + 2500 + 200
graph_idx_in = 869*700 + 2500*280 + 200*2000
graph_idx_out = 869*400 + 2500*100 + 200*600
graph_q_in = 0.85*4000 + 0.15*100000
graph_q_out = 0.85*119 + 0.15*300
graph_cost_q = cost(graph_q_in, graph_q_out)
print("GraphRAG index in/out", graph_idx_in, graph_idx_out, "cost", usd(cost(graph_idx_in, graph_idx_out)))
print("GraphRAG per query", usd(graph_cost_q), "150", usd(150*graph_cost_q))

print("\n=== STAGE TABLE ===")
tot_idx = 0; tot_q150 = 0
for k in [f"S{i}" for i in range(1,10)]:
    s = st[k]; ic, ii, io, icache = s["idx"]; qc, qi, qo = s["q"]
    icost = cost(ii, io, icache) if ic else 0
    qcost = cost(qi, qo)
    tot_idx += icost; tot_q150 += 150*qcost
    R[k] = dict(name=s["name"], idx_calls=ic, idx_in=ii, idx_out=io, idx_cached=icache,
                idx_usd=usd(icost), q_calls=qc, q_in=qi, q_out=qo, q_usd=usd(qcost),
                run150_usd=usd(150*qcost), run30_usd=usd(30*qcost),
                cum_idx_plus_150=usd(tot_idx + tot_q150),
                run150_in=150*qi, run150_out=150*qo, run30_in=30*qi, run30_out=30*qo)
    print(k, s["name"])
    print("  idx calls", ic, "in", round(ii), "out", round(io), "cached", round(icache), "usd", usd(icost))
    print("  q calls", round(qc,2), "in", round(qi,1), "out", round(qo,1), "usd", usd(qcost))
    print("  150 usd", usd(150*qcost), "30 usd", usd(30*qcost), "cum", usd(tot_idx+tot_q150))

qcost = {k: cost(st[k]["q"][1], st[k]["q"][2]) for k in st}
BUILT8 = [f"S{i}" for i in range(1, 9)]   # final eval: 8 built stages (S9 offline routing)
sum8_150 = 150 * sum(qcost[k] for k in BUILT8)
sum9_150 = 150 * sum(qcost[k] for k in st)
print("SUM 8 stages (S1-S8) 150x usd", usd(sum8_150))
print("SUM 9 stages 150x usd", usd(sum9_150))
print("TOTAL idx S5+S6", usd(cost(st["S5"]["idx"][1], st["S5"]["idx"][2], st["S5"]["idx"][3]) + cost(st["S6"]["idx"][1], st["S6"]["idx"][2])))

# dev loop: 5 iter x 30 q x 8 stages
dev = 5 * 30 * sum(qcost[k] for k in BUILT8)
print("DEV 5x30 8stages usd", usd(dev))

# S9 offline: only 150 router calls, answers reused from cached S1/S2/S7/S8 outputs
s9_router_only = 150 * cost(router_in, 8)
print("S9 router-only 150 usd", usd(s9_router_only))

# judge
j_in = T(JUDGE_SYS) + T(JUDGE_USER.format(question="", gold="", a="", b="")) + avg_q + 6 + 2*avg_out
j_out = 30
j_calls = 2400
j_cost = j_calls * cost(j_in, j_out)
print("JUDGE in/call", round(j_in,1), "out", j_out, "total usd", usd(j_cost))
print("JUDGE one-order 1200 usd", usd(1200*cost(j_in, j_out)))

# corpus prep contingency: re-label 200 docs if corpus rebuilt
corpus_prep = 200 * cost(1724 + 100, 10)
idx_s5 = cost(st["S5"]["idx"][1], st["S5"]["idx"][2], st["S5"]["idx"][3])
idx_s6 = cost(st["S6"]["idx"][1], st["S6"]["idx"][2])
lines = {
 "eval_gen": 120 * cost(3000, 400),
 "corpus_prep": corpus_prep,
 "s5_idx": idx_s5, "s6_idx": idx_s6,
 "dev": dev, "final150": sum8_150, "s9_offline": s9_router_only,
 "judge": j_cost,
}
lines["subtotal"] = sum(lines.values())
lines["buffer20"] = 0.20*lines["subtotal"]
lines["total"] = lines["subtotal"]*1.2
print("\n=== BUDGET ===")
for k,v in lines.items(): print(k, usd(v), "(" + str(round(v/10*100,2)) + "% of $10)")
print("headroom", usd(10-lines["total"]))

# scenarios (sustained on S1-S8; S9 ~0)
def q_sum(mult_in, mult_out, mult_calls=1.0):
    return sum(150*cost(st[k]["q"][1]*mult_in, st[k]["q"][2]*mult_out) for k in BUILT8)*mult_calls
scen = {}
scen["conservative"] = dict(dev=3*30*sum(qcost[k] for k in BUILT8)*0.8,
                            final=q_sum(0.85,0.75,1.0),
                            judge=2400*cost(j_in*0.85,j_out*0.8),
                            idx=(idx_s5*0.75+idx_s6*0.8)+corpus_prep*0.6+120*cost(3000,400))
scen["expected"] = dict(dev=dev, final=sum8_150, judge=j_cost,
                        idx=idx_s5+idx_s6+corpus_prep+120*cost(3000,400))
scen["worst"] = dict(dev=8*30*sum(qcost[k] for k in BUILT8)*1.4,
                     final=q_sum(1.3,1.6,1.15),
                     judge=2400*cost(j_in*1.3,j_out*1.5)*1.1,
                     idx=(idx_s5*1.6+idx_s6*1.5)+corpus_prep*1.2+120*cost(3000,400)*1.3)
print("\n=== SCENARIOS ===")
for k,v in scen.items():
    base = v["dev"]+v["final"]+v["judge"]+v["idx"]
    print(k, {kk:usd(vv) for kk,vv in v.items()}, "subtotal", usd(base), "with20%", usd(base*1.2))

json.dump(R, open(r"C:\Users\Ashit\Documents\GitHub\baby-rag-history\research\model.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
