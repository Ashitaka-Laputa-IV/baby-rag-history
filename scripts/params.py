"""Experiment parameters. Business layer only."""

from pathlib import Path

# data
CORPUS_PATH = Path("data/corpus.jsonl")
CHUNKS_PATH = Path("data/chunks.jsonl")
CHUNKS_MANIFEST_PATH = Path("data/manifest-chunks.json")
PERSIST_DIR = Path("data/chroma")
COLLECTION = "rag_m1"

# chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
ENCODING = "cl100k_base"
DELIM = ["。", "！", "？", "；", "\n"]
INCLUDE_DELIM = "prev"

# embedding
EMBED_MODEL = "BAAI/bge-m3"
EMBED_BASE_URL = "https://api.siliconflow.cn/v1"
EMBED_BATCH_SIZE = 32

# generation
LLM_MODEL = "deepseek-v4.1-flash"
LLM_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
LLM_MAX_OUTPUT_TOKENS = 1024
LLM_TIMEOUT_SECONDS = 90
LLM_MAX_RETRIES = 2

# retrieval / prompting
TOP_K = 5
MAX_CONTEXT_TOKENS = 4000
PROMPT_NAME = "m1"

# vector store
CHROMA_SPACE = "cosine"
CHROMA_BATCH_SIZE = 128

# evaluation
QUERIES_PATH = Path("eval/queries.jsonl")
RESULTS_DIR = Path("results/runs")

# pricing (opencode-go deepseek-v4.1-flash, USD per token)
PRICE_INPUT_PER_TOKEN = 0.15 / 1e6
PRICE_OUTPUT_PER_TOKEN = 0.60 / 1e6
