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
CHROMA_DIR = DATA / "chroma"
BM25_DIR = DATA / "bm25"

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


def load_settings(env: dict[str, str] | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    e = os.environ if env is None else env
    key = (e.get("OPENCODE_GO_API_KEY") or e.get("OPENCODE_API_KEY") or "").strip()
    if not key:
        raise ConfigError("OPENCODE_GO_API_KEY is not set (see .env.example)")
    session = (e.get("OPENCODE_SESSION_ID") or "").strip() or f"ses_{uuid.uuid4().hex}"
    return Settings(
        api_key=key,
        base_url=(e.get("OPENCODEGO_BASE_URL") or DEFAULT_BASE_URL).strip(),
        model=(e.get("OPENCODEGO_MODEL") or DEFAULT_MODEL).strip(),
        silicon_key=(e.get("SILICONFLOW_API_KEY") or "").strip(),
        silicon_url=(e.get("SILICONFLOW_BASE_URL") or DEFAULT_SILICON_URL).strip(),
        session_id=session,
        max_usd=float(e.get("BABYRAG_MAX_USD") or 10.0),
    )


def ensure_dirs() -> None:
    for path in (DATA, EVAL, RESULTS, CACHE_DB.parent, CHROMA_DIR, BM25_DIR):
        path.mkdir(parents=True, exist_ok=True)


def usd(tokens_in: int, tokens_out: int, cached_in: int = 0) -> float:
    return (tokens_in * PRICE_INPUT
            + cached_in * PRICE_CACHE_READ
            + tokens_out * PRICE_OUTPUT)
