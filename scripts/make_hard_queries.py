"""Generate a hard question set that cannot be solved by title matching.

Each question is written by the LLM from one chunk with three constraints:
no document title, no named entity from the chunk, and the answer must be an
explicit span in the chunk.  The evidence span is verified before writing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from babyrag.config import DATA, QUERIES_HARD, load_settings
from babyrag.llm import LLMClient

PROMPT = """你是中文维基百科的出题人。根据下面的片段写一道中文问题。

硬性要求：
1. 问题里不能出现文档标题、专有人名、地名或机构名。
2. 但必须带足够的限定信息（年份、国家/地区、职业、机构类型、事件类型），让读者能定位到唯一对象。
   错误示例：“这场战役开始于哪一天？”（太泛，无法定位）
   正确示例：“2012 年爆发于叙利亚、由反对派发起的这场战役开始于哪一天？”
3. 答案必须是片段中明确写出的事实（数字、年份、职位、地点、名称等）。
4. 问题要具体，不能问“这篇文章讲了什么”。
5. 只输出 JSON，不要 Markdown，不要解释：
   {{"question": "...", "answer": "...", "evidence": "片段中支持答案的原文 10-50 字"}}

文档标题：{title}
片段：
{chunk}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out", type=Path, default=QUERIES_HARD)
    args = ap.parse_args()

    settings = load_settings()
    llm = LLMClient(settings, max_output_tokens=300)
    docs = {d["doc_id"]: d for d in
            (json.loads(line) for line in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    chunks = [json.loads(line) for line in (DATA / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_theme: dict[str, list[dict]] = {}
    seen_doc: set[str] = set()
    rng = random.Random(7)
    rng.shuffle(chunks)
    for chunk in chunks:
        if chunk["doc_id"] in seen_doc:
            continue
        # prefer chunks with digits or long factual text
        if len(chunk["text"]) < 300:
            continue
        by_theme.setdefault(chunk["theme"], []).append(chunk)
    per_theme = max(1, args.count // max(1, len(by_theme)))
    picked: list[dict] = []
    for theme, items in by_theme.items():
        picked.extend(items[:per_theme])
    picked = picked[: args.count]

    def parse(text: str) -> dict | None:
        match = __import__("re").search(r"\{.*\}", text, __import__("re").S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    rows: list[dict] = []
    for i, chunk in enumerate(picked, start=1):
        title = docs[chunk["doc_id"]]["title"]
        body = None
        for attempt in range(2):
            suffix = "" if attempt == 0 else "\n\n上一次输出不是合法 JSON。只输出 JSON，不要任何其他文字。"
            result = llm.invoke(PROMPT.format(title=title, chunk=chunk["text"][:1800]) + suffix,
                                stage="hard_query", prompt_version=f"v{attempt + 1}",
                                system="你只输出合法 JSON。")
            body = parse(result.text)
            if body:
                break
        if not body:
            print(f"skip {chunk['chunk_id']}: bad JSON")
            continue
        question = str(body.get("question", "")).strip()
        answer = str(body.get("answer", "")).strip()
        evidence = str(body.get("evidence", "")).strip()
        if not question or not answer or not evidence:
            print(f"skip {chunk['chunk_id']}: empty field")
            continue
        if title in question or evidence not in chunk["text"]:
            print(f"skip {chunk['chunk_id']}: title leak or evidence mismatch")
            continue
        rows.append({
            "qid": f"h{i:03d}", "type": "hard_single", "theme": chunk["theme"],
            "question": question, "gold_answer": answer,
            "gold_doc_ids": [chunk["doc_id"]], "evidence": evidence,
            "should_abstain": False, "synthetic": True,
        })
        print(f"{i}/{len(picked)} {question}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
