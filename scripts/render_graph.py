"""Render the actual compiled graph to docs/graph.png and docs/graph.html.

Structural build only: no node executes, so no API key or index is needed.
The PNG comes from LangGraph's Mermaid renderer; the HTML embeds it as base64
so the page is self-contained.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

from babyrag.graph import build_graph

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main() -> int:
    graph = build_graph(None, None)  # type: ignore[arg-type]
    view = graph.get_graph()
    mermaid = view.draw_mermaid()
    nodes, edges = len(view.nodes), len(view.edges)

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "graph.mmd").write_text(mermaid, encoding="utf-8")

    png = b""
    try:
        png = view.draw_mermaid_png()
        (DOCS / "graph.png").write_bytes(png)
        print(f"PNG: {len(png)} bytes, {nodes} nodes, {edges} edges")
    except Exception as exc:  # offline renderer unavailable
        print(f"PNG render failed ({type(exc).__name__}: {exc}); writing HTML only")

    img = (f'<img alt="LangGraph 结构：{nodes} 个节点，{edges} 条边" '
           f'src="data:image/png;base64,{base64.b64encode(png).decode()}">'
           if png else "<p>PNG 未生成；请看下方 Mermaid 源。</p>")

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>baby-rag-history · LangGraph 结构</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          margin: 2rem auto; max-width: 1100px; padding: 0 1.2rem; color: #222; }}
  h1 {{ font-size: 1.5rem; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; background: #fff; }}
  pre {{ background: #f6f8fa; padding: 1rem; overflow: auto; font-size: .82rem; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
  .meta {{ color: #666; }}
</style>
</head>
<body>
<h1>baby-rag-history · LangGraph 结构</h1>
<p class="meta">由 <code>scripts/render_graph.py</code> 从 <code>build_graph()</code> 生成：
{nodes} 个节点，{edges} 条边。只做结构编译，不执行任何节点。</p>
{img}
<h2>Mermaid 源</h2>
<details><summary>展开</summary><pre>{html.escape(mermaid)}</pre></details>
</body>
</html>
"""
    (DOCS / "graph.html").write_text(page, encoding="utf-8")
    print(f"wrote {DOCS / 'graph.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
