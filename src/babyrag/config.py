"""Environment, paths and prices. No business logic."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = ROOT / "eval"
RESULTS = ROOT / "results"
CACHE_DB = DATA / "cache" / "llm.sqlite3"

CORPUS = DATA / "corpus.jsonl"
CHUNKS = DATA / "chunks.jsonl"
CONTEXTUAL = DATA / "contextual.jsonl"
RAPTOR_NODES = DATA / "raptor.jsonl"
QUERIES = EVAL / "queries.jsonl"
QUERIES_HARD = EVAL / "queries-hard.jsonl"
CHROMA_DIR = DATA / "chroma"
BM25_DIR = DATA / "bm25"


def variant_paths(variant: str) -> dict[str, Path | str]:
    """Resolve file paths and Chroma collections for a corpus variant.

    The three variants share one layout; only the base directory and the two
    suffixes differ.
    """
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

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_SILICON_URL = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

PRICE_INPUT = 0.15 / 1e6
PRICE_OUTPUT = 0.60 / 1e6
PRICE_CACHE_READ = 0.003 / 1e6

MAX_CONTEXT_TOKENS = 4000
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
ENCODING = "cl100k_base"
DELIM = ["。", "！", "？", "；", "\n"]
INCLUDE_DELIM = "prev"


class ConfigError(RuntimeError):
    """Raised when a required key is missing."""


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


def load_settings(env: dict[str, str] | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    e = os.environ if env is None else env
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

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        silicon_key=silicon_key,
        silicon_url=silicon_url,
        session_id=session,
        max_usd=float(e.get("BABYRAG_MAX_USD") or 10.0),
        variant=(e.get("BABYRAG_VARIANT") or "original").strip(),
        provider=provider,
    )


def ensure_dirs() -> None:
    for path in (DATA, EVAL, RESULTS, CACHE_DB.parent, CHROMA_DIR, BM25_DIR):
        path.mkdir(parents=True, exist_ok=True)


def usd(tokens_in: int, tokens_out: int, cached_in: int = 0) -> float:
    return (tokens_in * PRICE_INPUT
            + cached_in * PRICE_CACHE_READ
            + tokens_out * PRICE_OUTPUT)
