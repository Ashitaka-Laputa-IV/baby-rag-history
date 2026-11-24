# 项目网页

线上地址：<https://ashitaka-laputa-iv.github.io/baby-rag-history/>

这是 baby-rag-history 的单页展示站，结构参考了 MiniCode 的 `site/`：纯静态 HTML/CSS/JS，没有构建步骤。

## 本地预览

直接打开 `site/index.html`，或在仓库根目录启动一个静态服务器：

```bash
uv run python -m http.server 8000 --directory site
```

然后访问 <http://localhost:8000/>。

## 部署

`.github/workflows/pages.yml` 使用 GitHub Pages Actions：

- 手动运行 `Deploy project site`；
- 或推送一个 `site-*` tag，例如 `site-2026-11-24`。

工作流会把 `site/` 作为静态站点上传。架构图是 `site/index.html` + `site/styles.css`
手工构建的 HTML 流程图，不再依赖 `docs/graph.png` 或 Mermaid 截图。
