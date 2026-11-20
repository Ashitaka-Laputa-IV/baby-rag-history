"""The ladder: each milestone is one ordered plan, not one graph."""

from __future__ import annotations

RUNGS: dict[str, list[str]] = {
    "r0_dense": ["retrieve_dense", "generate"],
    "r1_hybrid": ["retrieve_hybrid", "fuse_rrf", "generate"],
    "r2_rerank": ["retrieve_hybrid", "fuse_rrf", "rerank", "generate"],
    "r3_hyde": ["hyde", "retrieve_hybrid", "fuse_rrf", "rerank", "generate"],
    "r4_multiquery": ["rewrite_multi_query", "retrieve_multi", "fuse_rrf", "rerank", "generate"],
    "r5_contextual": ["retrieve_contextual", "fuse_rrf", "rerank", "generate"],
    "r6_raptor": ["retrieve_raptor", "rerank", "generate"],
    "r7_crag": ["retrieve_hybrid", "fuse_rrf", "rerank", "grade", "generate"],
    "r8_agentic": ["agentic", "verify", "generate"],
}

ADAPTIVE = "adaptive"
ALL_RUNGS = list(RUNGS) + [ADAPTIVE]

LABELS: dict[str, str] = {
    "r0_dense": "Naive RAG (dense top-5)",
    "r1_hybrid": "Hybrid BM25 + dense + RRF",
    "r2_rerank": "Hybrid + bge-reranker",
    "r3_hyde": "HyDE hypothetical document",
    "r4_multiquery": "Multi-query / RAG-Fusion",
    "r5_contextual": "Contextual Retrieval",
    "r6_raptor": "RAPTOR summary tree",
    "r7_crag": "CRAG grade + abstain",
    "r8_agentic": "Agentic ReAct",
    "adaptive": "Adaptive router",
}

PAPERS: dict[str, str] = {
    "r0_dense": "Lewis et al. 2020, arXiv:2005.11401",
    "r1_hybrid": "Cormack et al. 2009 RRF; Robertson & Zaragoza 2009 BM25",
    "r2_rerank": "Nogueira & Cho 2019, arXiv:1901.04085",
    "r3_hyde": "Gao et al. 2022, arXiv:2212.10496",
    "r4_multiquery": "Rackauckas 2024, arXiv:2402.03367",
    "r5_contextual": "Anthropic 2024-09, Contextual Retrieval",
    "r6_raptor": "Sarthi et al. 2024, arXiv:2401.18059",
    "r7_crag": "Yan et al. 2024, arXiv:2401.15884; Asai et al. 2023, arXiv:2310.11511",
    "r8_agentic": "Yao et al. 2022, arXiv:2210.03629",
    "adaptive": "Jeong et al. 2024, arXiv:2403.14403",
}


def adaptive_plan(question: str, qtype: str | None = None) -> tuple[str, str]:
    """Cheap rule-first router; the LLM is only a fallback in the node."""
    if qtype == "multi":
        return "r8_agentic", "type=multi"
    if qtype in {"none", "no_answer"}:
        return "r7_crag", "type=no_answer"
    if qtype == "global":
        return "r6_raptor", "type=global"
    if len(question) > 60:
        return "r4_multiquery", "long_question"
    return "r1_hybrid", "default"
