"""One LangGraph state machine.  Milestones are plans, not separate graphs."""

from __future__ import annotations

import time
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .config import MAX_CONTEXT_TOKENS
from .llm import LLMClient
from .prompts import (ADAPTIVE, CONTEXTUAL, GENERATE, GRADE, HYDE, JUDGE,
                      MULTI_QUERY, SYSTEM_GROUNDED, parse_queries,
                      parse_relevant, parse_route)
from .retrieval import (Services, build_context, extract_citations, fuse_rrf,
                        rerank_hits)
from .rungs import RUNGS, adaptive_plan
from .types import Chunk, NodeRecord, RAGState, add_cost

AGENTIC = """你要回答一个中文问题。你可以先搜索维基语料。

问题：{question}

历史：
{history}

只输出一行：
- 需要搜索时：SEARCH: <检索词>
- 信息足够时：FINAL: <最终答案>
"""

STAGES = [
    "retrieve_dense", "retrieve_hybrid", "retrieve_contextual", "retrieve_multi",
    "retrieve_raptor", "hyde", "rewrite_multi_query", "fuse_rrf", "rerank",
    "expand_context", "grade", "agentic", "verify", "generate",
]


class Runner:
    def __init__(self, services: Services, llm: LLMClient) -> None:
        self.services = services
        self.llm = llm

    # ---------------------------------------------------------------- helpers
    def _query(self, state: RAGState) -> str:
        variants = state.get("query_variants") or []
        return variants[-1] if variants else state["question"]

    def _record(self, node: str, result: Any, note: str = "") -> NodeRecord:
        return self.llm.record(result, node, note)

    def _hybrid(self, query: str) -> tuple[list[Chunk], list[Chunk]]:
        dense = self.services.dense.search(query, k=10)
        bm25 = self.services.bm25.search(query, k=10)
        return dense, bm25

    # ------------------------------------------------------------------ nodes
    def plan(self, state: RAGState) -> dict:
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

    def dispatch(self, state: RAGState) -> dict:
        return {}

    def route(self, state: RAGState) -> str:
        plan = state.get("plan") or []
        cursor = int(state.get("cursor", 0))
        return plan[cursor] if cursor < len(plan) else "finalize"

    def retrieve_dense(self, state: RAGState) -> dict:
        hits = self.services.dense.search(self._query(state), k=10)
        return {"hits": hits, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "retrieve_dense", "detail": f"{len(hits)} hits"}]}

    def retrieve_hybrid(self, state: RAGState) -> dict:
        dense, bm25 = self._hybrid(self._query(state))
        return {"dense_hits": dense, "bm25_hits": bm25, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "retrieve_hybrid",
                                "detail": f"dense={len(dense)} bm25={len(bm25)}"}]}

    def retrieve_contextual(self, state: RAGState) -> dict:
        query = self._query(state)
        dense = self.services.contextual_dense.search(query, k=10, source="dense_contextual")
        bm25 = self.services.contextual_bm25.search(query, k=10)
        return {"dense_hits": dense, "bm25_hits": bm25, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "retrieve_contextual",
                                "detail": f"dense={len(dense)} bm25={len(bm25)}"}]}

    def retrieve_multi(self, state: RAGState) -> dict:
        variants = state.get("query_variants") or [state["question"]]
        lists: list[list[Chunk]] = []
        for query in variants:
            dense, bm25 = self._hybrid(query)
            lists.append(fuse_rrf(dense, bm25, top_n=10))
        return {"multi_hits": lists, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "retrieve_multi", "detail": f"{len(lists)} queries"}]}

    def fuse_rrf(self, state: RAGState) -> dict:
        multi = state.get("multi_hits") or []
        dense = state.get("dense_hits") or []
        bm25 = state.get("bm25_hits") or []
        if multi:
            hits = fuse_rrf(*multi, top_n=20)
        else:
            hits = fuse_rrf(dense, bm25, top_n=20)
        return {"hits": hits, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "fuse_rrf", "detail": f"{len(hits)} fused"}]}

    def rerank(self, state: RAGState) -> dict:
        hits = rerank_hits(self.services.settings, self._query(state),
                           state.get("hits") or [], top_n=5)
        if not hits:
            hits = (state.get("hits") or [])[:5]
        return {"hits": hits, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "rerank", "detail": f"top {len(hits)}"}]}

    def hyde(self, state: RAGState) -> dict:
        result = self.llm.invoke(HYDE.format(question=state["question"]),
                                 stage="hyde", system="你是一位中文维基百科作者。")
        return {"query_variants": [result.text.strip() or state["question"]],
                "cursor": state["cursor"] + 1,
                "node_records": [self._record("hyde", result)],
                "trajectory": [{"node": "hyde", "detail": f"{len(result.text)} chars"}]}

    def rewrite_multi_query(self, state: RAGState) -> dict:
        result = self.llm.invoke(MULTI_QUERY.format(question=state["question"]),
                                 stage="multi_query")
        variants = [state["question"]] + parse_queries(result.text, 3)
        return {"query_variants": variants, "cursor": state["cursor"] + 1,
                "node_records": [self._record("multi_query", result)],
                "trajectory": [{"node": "multi_query", "detail": f"{len(variants)} variants"}]}

    def expand_context(self, state: RAGState) -> dict:
        out: list[Chunk] = []
        for hit in state.get("hits") or []:
            extra = self.services.contextual.get(hit["chunk_id"])
            if extra:
                hit = dict(hit)
                hit["context"] = extra.get("prefix", "")
                hit["text"] = f"{extra.get('prefix', '')}\n{hit['text']}"
            out.append(hit)  # type: ignore[arg-type]
        return {"hits": out, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "expand_context", "detail": f"{len(out)} expanded"}]}

    def retrieve_raptor(self, state: RAGState) -> dict:
        # Collapsed tree: leaves and summary nodes are searched together, then
        # fused. Summary-only retrieval loses every factual detail.
        query = self._query(state)
        leaves = self.services.dense.search(query, k=10, source="raptor_leaf")
        try:
            summaries = self.services.raptor.search(query, k=10, source="raptor")
        except Exception:
            summaries = []
        hits = fuse_rrf(leaves, summaries, top_n=20) if summaries else leaves
        return {"hits": hits, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "retrieve_raptor",
                                "detail": f"{len(leaves)} leaves + {len(summaries)} summaries"}]}

    def grade(self, state: RAGState) -> dict:
        hits = (state.get("hits") or [])[:6]
        if not hits:
            return {"grade": "incorrect", "cursor": state["cursor"] + 1,
                    "trajectory": [{"node": "grade", "detail": "no hits"}]}
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
                    "cursor": state["cursor"] + 1, "node_records": records,
                    "trajectory": [{"node": "grade", "detail": f"{verdict} -> rewrite"}]}
        if relevant:
            kept = [hits[i - 1] for i in relevant if 1 <= i <= len(hits)]
            hits = kept
        return {"hits": hits, "grade": verdict, "cursor": state["cursor"] + 1,
                "node_records": records,
                "trajectory": [{"node": "grade", "detail": verdict}]}

    def agentic(self, state: RAGState) -> dict:
        collected: dict[str, Chunk] = {}
        history: list[str] = []
        records: list[NodeRecord] = []
        # Baseline retrieval first: the first search should always be the question
        # itself, otherwise an early rewritten query can crowd out the gold chunk.
        for hit in fuse_rrf(*self._hybrid(state["question"]), top_n=5):
            collected.setdefault(hit["chunk_id"], hit)
        for step in range(4):
            prompt = AGENTIC.format(question=state["question"],
                                    history="\n".join(history) or "(无)")
            result = self.llm.invoke(prompt, stage="agentic", extra_key=str(step))
            records.append(self._record("agentic", result, f"step {step + 1}"))
            text = result.text.strip()
            if text.startswith("FINAL:") or "FINAL:" in text:
                break
            query = text.split("SEARCH:", 1)[-1].strip().splitlines()[0] if "SEARCH:" in text else text
            if not query:
                break
            hits = fuse_rrf(*self._hybrid(query), top_n=5)
            for hit in hits:
                collected.setdefault(hit["chunk_id"], hit)
            history.append(f"SEARCH: {query}\nOBSERVATION: " +
                           " ".join(h["text"][:120] for h in hits[:3]))
        if not collected:
            dense, bm25 = self._hybrid(state["question"])
            for hit in fuse_rrf(dense, bm25, top_n=8):
                collected.setdefault(hit["chunk_id"], hit)
        hits = list(collected.values())
        return {"hits": hits, "cursor": state["cursor"] + 1, "node_records": records,
                "trajectory": [{"node": "agentic", "detail": f"{len(hits)} collected"}]}

    def generate(self, state: RAGState) -> dict:
        # Fairness rule: every rung gets at most five chunks in the final context.
        hits = (state.get("hits") or [])[:5]
        context, kept = build_context(hits, services=self.services,
                                      budget=MAX_CONTEXT_TOKENS)
        if not kept:
            return {"answer": "无法回答", "abstain": True, "hits": [],
                    "cursor": state["cursor"] + 1,
                    "trajectory": [{"node": "generate", "detail": "no context"}]}
        result = self.llm.invoke(GENERATE.format(question=state["question"], context=context),
                                 stage="generate", system=SYSTEM_GROUNDED)
        answer = result.text.strip()
        abstain = answer.startswith("无法回答") and len(answer) < 20
        return {"answer": answer, "abstain": abstain, "hits": kept,
                "citations": extract_citations(answer, kept),
                "cursor": state["cursor"] + 1,
                "node_records": [self._record("generate", result)],
                "trajectory": [{"node": "generate", "detail": f"{len(answer)} chars"}]}

    def verify(self, state: RAGState) -> dict:
        answer = state.get("answer") or ""
        abstain = state.get("abstain", False) or answer.strip() == "无法回答"
        return {"abstain": abstain, "cursor": state["cursor"] + 1,
                "trajectory": [{"node": "verify", "detail": f"abstain={abstain}"}]}

    def finalize(self, state: RAGState) -> dict:
        records = state.get("node_records") or []
        return {"cost": add_cost(records),
                "trajectory": [{"node": "finalize", "detail": f"${add_cost(records)['usd']:.4f}"}]}


def build_graph(services: Services, llm: LLMClient):
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


def run_one(graph, *, question: str, rung: str = "r0_dense",
            mode: str = "ladder", qid: str = "", qtype: str = "") -> dict:
    state: RAGState = {
        "qid": qid, "question": question, "rung": rung, "mode": mode,
        "qtype": qtype, "retries": 0, "node_records": [], "trajectory": [],
    }
    started = time.perf_counter()
    result = graph.invoke(state, config={"recursion_limit": 60})
    elapsed = int((time.perf_counter() - started) * 1000)
    cost = dict(result.get("cost") or {})
    cost["latency_ms"] = elapsed
    return {
        "qid": qid, "rung": rung, "question": question,
        "answer": result.get("answer", ""), "abstain": bool(result.get("abstain", False)),
        "citations": result.get("citations", []),
        "hits": [{"chunk_id": h.get("chunk_id", ""), "doc_id": h.get("doc_id", ""),
                  "doc_ids": [x for x in (h.get("doc_ids") or []) if x],
                  "text": h.get("text", ""),
                  "score": h.get("score", 0.0), "source": h.get("source", "")}
                 for h in result.get("hits", [])],
        "plan": result.get("plan", []),
        "route_reason": result.get("route_reason", ""),
        "cost": cost,
        "node_records": result.get("node_records", []),
    }
