"""SQLite key-value cache for every paid LLM call.

Key = sha256(stage | prompt_version | model | normalized input).  Cache hits
cost nothing, which is what keeps repeated dev runs cheap.

The database is opened by several evaluation threads at once, so initialisation
and writes retry on SQLite's "database is locked" instead of failing a query.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _retry(fn: Callable[[], T], *, attempts: int = 8) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            last = exc
            time.sleep(0.15 * (i + 1))
    raise last  # pragma: no cover


class Cache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self._db.execute("PRAGMA busy_timeout=30000")
        mode = self._db.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            try:
                self._db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass
        _retry(lambda: self._db.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            " key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " model TEXT, created_at REAL NOT NULL)"))
        _retry(self._db.commit)

    def get(self, key: str) -> str | None:
        row = _retry(lambda: self._db.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)).fetchone())
        return row[0] if row else None

    def put(self, key: str, value: str, model: str = "") -> None:
        _retry(lambda: self._db.execute(
            "INSERT OR REPLACE INTO kv (key, value, model, created_at) VALUES (?, ?, ?, ?)",
            (key, value, model, time.time())))
        _retry(self._db.commit)

    def count(self) -> int:
        return int(_retry(lambda: self._db.execute("SELECT COUNT(*) FROM kv").fetchone())[0])

    def close(self) -> None:
        self._db.close()


class JsonCache:
    """A tiny JSON-file cache for structured offline artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
