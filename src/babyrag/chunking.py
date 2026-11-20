"""Chunking core: a thin wrapper around chonkie's SentenceChunker.

No experiment parameters live here. Callers pass chunk size, overlap, encoding
and delimiters; the business layer (scripts/build_chunks.py) owns those values.
Pure logic only: no file IO, no printing, no argument parsing.
"""

from __future__ import annotations

import tiktoken
from chonkie import SentenceChunker


class DocumentChunker:
    """Wraps SentenceChunker and turns documents into chunk records."""

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        encoding: str,
        delim: list[str],
        include_delim: str | None = "prev",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = encoding
        self._encoder = tiktoken.get_encoding(encoding)
        self._chunker = SentenceChunker(
            tokenizer=self._encoder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            delim=delim,
            include_delim=include_delim,
        )

    def chunk_document(self, doc: dict) -> list[dict]:
        """Split one corpus document into chunk records with stable ids."""
        text = doc["text"]
        if not text:
            return []

        rows: list[dict] = []
        for seq, chunk in enumerate(self._chunker.chunk(text)):
            rows.append(
                {
                    "chunk_id": f"{doc['doc_id']}:{seq:03d}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "theme": doc["theme"],
                    "seq": seq,
                    "text": chunk.text,
                    "start": chunk.start_index,
                    "end": chunk.end_index,
                    "tokens": chunk.token_count,
                    "chars": chunk.end_index - chunk.start_index,
                }
            )
        return rows
