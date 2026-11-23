"""Fast, network-free tests for the pieces that decide correctness."""

from __future__ import annotations

from babyrag.config import variant_paths
from babyrag.prompts import parse_queries, parse_relevant, parse_route, parse_score
from babyrag.retrieval import fuse_rrf
from babyrag.rungs import RUNGS, adaptive_plan
from babyrag.types import Chunk


def chunk(cid: str, score: float = 1.0) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid.split(":")[0], doc_ids=[cid.split(":")[0]],
                 text=cid, score=score, rank=1, source="test")


def test_rrf_prefers_agreement() -> None:
    a = [chunk("a"), chunk("b"), chunk("c")]
    b = [chunk("b"), chunk("a"), chunk("d")]
    fused = fuse_rrf(a, b, top_n=4)
    assert [h["chunk_id"] for h in fused][:2] == ["a", "b"]
    assert len(fused) == 4


def test_parse_relevant_is_bounded() -> None:
    rel, verdict = parse_relevant('{"relevant": [1, 3, 99], "verdict": "correct"}', 3)
    assert rel == [1, 3]
    assert verdict == "correct"


def test_parse_queries_trims_noise() -> None:
    assert parse_queries("1. 甲\n- 乙\n丙\n丁") == ["1. 甲", "乙", "丙"]


def test_parse_score() -> None:
    assert parse_score('{"score": 1, "reason": "ok"}') == 1
    assert parse_score('{"score": 0, "reason": "no"}') == 0


def test_parse_route() -> None:
    assert parse_route("multi") == "multi"
    assert parse_route("this is GLOBAL") == "global"
    assert parse_route("?") == "simple"


def test_every_rung_ends_with_generate() -> None:
    for name, plan in RUNGS.items():
        assert plan[-1] == "generate", name
        assert "generate" in plan


def test_adaptive_rules() -> None:
    assert adaptive_plan("q", "multi")[0] == "r8_agentic"
    assert adaptive_plan("q", "none")[0] == "r7_crag"
    assert adaptive_plan("q", "global")[0] == "r6_raptor"
    assert adaptive_plan("短问题", "single")[0] == "r1_hybrid"


def test_variant_paths_share_one_shape() -> None:
    original = variant_paths("original")
    masked = variant_paths("masked")
    public = variant_paths("public")
    assert original["corpus"].name == "corpus.jsonl"
    assert masked["corpus"].name == "corpus.masked.jsonl"
    assert public["corpus"].parent.name == "public"
    assert original["dense"] == "chunks"
    assert masked["dense"] == "chunks_masked"
    assert public["dense"] == "chunks_public"
