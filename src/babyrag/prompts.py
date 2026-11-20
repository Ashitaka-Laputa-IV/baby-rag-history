"""Prompt templates and small parsers. Version them here, not in nodes."""

from __future__ import annotations

import json
import re

PROMPT_VERSION = "v1"

SYSTEM_GROUNDED = (
    "只依据给定资料回答问题。每个结论后用 [编号] 标注来源，例如 [1]。"
    "资料不足时只回答“无法回答”，不要编造，也不要引用未提供的资料。"
)

GENERATE = """问题：{question}

资料：
{context}
"""

HYDE = """请写一段 150–250 字、维基百科风格的中文段落，内容要像是在回答下面这个问题。
不要解释，不要加标题，直接写段落。

问题：{question}
"""

MULTI_QUERY = """把下面的问题改写成 3 个语义相同但措辞不同的中文检索查询。
每行一个，不要编号，不要解释。

问题：{question}
"""

CONTEXTUAL = """下面是维基百科文章《{title}》的全文（或节选）与其中一个片段。
请写一句 50–100 token 的中文说明，指出该片段在全文中的位置与主题，使其脱离全文也能被理解。
只输出这句说明，不要复述原文。

文章：
{doc}

片段：
{chunk}
"""

GRADE = """判断下面每个片段是否包含回答问题的信息。
问题：{question}

片段：
{chunks}

只输出 JSON：{{"relevant": [相关片段的序号], "verdict": "correct|ambiguous|incorrect"}}
relevant 用 1-based 序号。全部不相关时 relevant 为空数组，verdict 为 incorrect。
"""

ADAPTIVE = """把问题分成三类之一：
simple（单跳事实）、multi（多跳或比较）、global（需要跨多篇总结）。只输出类别名。

问题：{question}
"""

JUDGE = """你是严格的评测员。判断【回答】是否与【参考答案】一致，且没有编造。
只输出 JSON：{{"score": 1, "reason": "..."}}。score=1 表示正确，0 表示错误或拒答错误。
允许措辞不同；只比较事实。

问题：{question}
参考答案：{gold}
回答：{answer}
"""


def parse_relevant(text: str, n: int) -> tuple[list[int], str]:
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return list(range(1, n + 1)), "ambiguous"
    try:
        body = json.loads(match.group(0))
        rel = [int(x) for x in body.get("relevant", []) if 1 <= int(x) <= n]
        verdict = str(body.get("verdict", "ambiguous"))
        if verdict not in {"correct", "ambiguous", "incorrect"}:
            verdict = "ambiguous"
        return rel, verdict
    except (ValueError, TypeError):
        return list(range(1, n + 1)), "ambiguous"


def parse_score(text: str) -> int:
    match = re.search(r"\{.*?\}", text, re.S)
    if match:
        try:
            return 1 if int(json.loads(match.group(0)).get("score", 0)) == 1 else 0
        except (ValueError, TypeError):
            pass
    return 1 if "score" in text and "1" in text else 0


def parse_queries(text: str, limit: int = 3) -> list[str]:
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln][:limit]


def parse_route(text: str) -> str:
    low = text.strip().lower()
    for name in ("global", "multi", "simple"):
        if name in low:
            return name
    return "simple"
