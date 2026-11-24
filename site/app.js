(() => {
  "use strict";

  const rungs = {
    r0_dense: {
      id: "r0_dense",
      index: "r0 / 09",
      tab: "r0",
      name: "Naive RAG",
      sub: "只做 dense 检索，拿到 top-5 后直接生成。",
      change: "没有融合、没有精排、没有查询改写；这是整条梯子的基线。",
      plan: ["retrieve_dense", "generate"],
      metrics: { acc: "88.9%", evidence: "74.1%", mrr: "0.747", cost: "$0.00030" },
      score: 79.2,
      paper: "Lewis et al. 2020",
      paperUrl: "https://arxiv.org/abs/2005.11401",
      accent: "#ff8a8a",
      soft: "#fff0f0",
    },
    r1_hybrid: {
      id: "r1_hybrid",
      index: "r1 / 09",
      tab: "r1",
      name: "Hybrid + RRF",
      sub: "dense 加 BM25，两路排名用 RRF 融合。",
      change: "加入关键词检索一路；RRF 只看名次，避免两路分数量纲不同。",
      plan: ["retrieve_hybrid", "fuse_rrf", "generate"],
      metrics: { acc: "88.9%", evidence: "77.8%", mrr: "0.762", cost: "$0.00030" },
      score: 80.6,
      paper: "Cormack et al. 2009",
      paperUrl: "https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf",
      accent: "#ffb05c",
      soft: "#fff0dc",
    },
    r2_rerank: {
      id: "r2_rerank",
      index: "r2 / 09",
      tab: "r2",
      name: "bge-reranker",
      sub: "先召回 20 条，再让 reranker 挑最终 5 条。",
      change: "在融合之后加一层精排；交叉编码器能看到 query 和文档的交互。",
      plan: ["retrieve_hybrid", "fuse_rrf", "rerank", "generate"],
      metrics: { acc: "96.3%", evidence: "77.8%", mrr: "0.783", cost: "$0.00030" },
      score: 84.6,
      paper: "Nogueira & Cho 2019",
      paperUrl: "https://arxiv.org/abs/1901.04085",
      accent: "#e8b82e",
      soft: "#fff7d6",
    },
    r3_hyde: {
      id: "r3_hyde",
      index: "r3 / 09",
      tab: "r3",
      name: "HyDE",
      sub: "让模型先写一段“假答案”，再用它去检索。",
      change: "把短问题变成一段维基风格文本；可能补足语义，也可能把正确卡片挤掉。",
      plan: ["hyde", "retrieve_hybrid", "fuse_rrf", "rerank", "generate"],
      metrics: { acc: "96.3%", evidence: "48.1%", mrr: "0.628", cost: "$0.00081" },
      score: 69.5,
      paper: "Gao et al. 2022",
      paperUrl: "https://arxiv.org/abs/2212.10496",
      accent: "#5aa9f5",
      soft: "#e3f3ff",
    },
    r4_multiquery: {
      id: "r4_multiquery",
      index: "r4 / 09",
      tab: "r4",
      name: "Multi-Query",
      sub: "把问题改写成 3 个查询，分别检索后再融合。",
      change: "扩大查询覆盖面；每个变体内部先做一次 RRF，再做跨变体融合。",
      plan: ["rewrite_multi_query", "retrieve_multi", "fuse_rrf", "rerank", "generate"],
      metrics: { acc: "92.6%", evidence: "81.5%", mrr: "0.789", cost: "$0.00046" },
      score: 82.7,
      paper: "Rackauckas 2024",
      paperUrl: "https://arxiv.org/abs/2402.03367",
      accent: "#8d86ff",
      soft: "#eceaff",
    },
    r5_contextual: {
      id: "r5_contextual",
      index: "r5 / 09",
      tab: "r5",
      name: "Contextual Retrieval",
      sub: "给每个 chunk 补一句“它在全文里的位置”，再检索。",
      change: "索引期让模型生成定位前缀；检索和生成流程不变，改变的是卡片本身的语境。",
      plan: ["retrieve_contextual", "fuse_rrf", "rerank", "generate"],
      metrics: { acc: "100.0%", evidence: "85.2%", mrr: "0.833", cost: "$0.00031" },
      score: 89.5,
      paper: "Anthropic 2024",
      paperUrl: "https://www.anthropic.com/news/contextual-retrieval",
      accent: "#31b98f",
      soft: "#dcfaef",
    },
    r6_raptor: {
      id: "r6_raptor",
      index: "r6 / 09",
      tab: "r6",
      name: "RAPTOR",
      sub: "叶子和摘要树一起检索，再做跨层融合。",
      change: "增加摘要节点，让跨文档的概括性问题也能命中；不是只查摘要。",
      plan: ["retrieve_raptor", "rerank", "generate"],
      metrics: { acc: "96.3%", evidence: "81.5%", mrr: "0.800", cost: "$0.00030" },
      score: 86.0,
      paper: "Sarthi et al. 2024",
      paperUrl: "https://arxiv.org/abs/2401.18059",
      accent: "#f083b6",
      soft: "#ffe6f2",
    },
    r7_crag: {
      id: "r7_crag",
      index: "r7 / 09",
      tab: "r7",
      name: "CRAG",
      sub: "先判断资料相关性；不对就改写问题，再查一次。",
      change: "在生成前加一次自检和最多一次重查；弃答仍然由最后的 generate 决定。",
      plan: ["retrieve_hybrid", "fuse_rrf", "rerank", "grade", "generate"],
      metrics: { acc: "92.6%", evidence: "81.5%", mrr: "0.783", cost: "$0.00065" },
      score: 81.1,
      paper: "Yan et al. 2024",
      paperUrl: "https://arxiv.org/abs/2401.15884",
      accent: "#2fb4c4",
      soft: "#defafa",
    },
    r8_agentic: {
      id: "r8_agentic",
      index: "r8 / 09",
      tab: "r8",
      name: "Agentic ReAct",
      sub: "最多 4 步 SEARCH → OBSERVATION，自己决定何时停止。",
      change: "把检索变成可重复调用的工具；先跑原始问题占住 baseline，再逐步补充。",
      plan: ["agentic", "verify", "generate"],
      metrics: { acc: "88.9%", evidence: "77.8%", mrr: "0.762", cost: "$0.00086" },
      score: 76.5,
      paper: "Yao et al. 2022",
      paperUrl: "https://arxiv.org/abs/2210.03629",
      accent: "#8b6fe8",
      soft: "#f0eaff",
    },
    adaptive: {
      id: "adaptive",
      index: "A / 09",
      tab: "A",
      name: "Adaptive Router",
      sub: "按题型和长度选一档：规则优先，不靠 LLM 猜。",
      change: "multi → r8；no-answer → r7；global → r6；长问题 → r4；其余默认 r1。",
      plan: ["adaptive_plan()", "RUNGS[selected]", "…"],
      metrics: { acc: "88.9%", evidence: "81.5%", mrr: "0.794", cost: "$0.00045" },
      score: 81.1,
      paper: "Jeong et al. 2024",
      paperUrl: "https://arxiv.org/abs/2403.14403",
      accent: "#45435c",
      soft: "#efeff4",
    },
  };

  const resultSets = {
    original: {
      title: "原题 30 · 原索引",
      note: "r5 Contextual Retrieval 最高：100% 可答 acc、85.2% evidence R@1。",
      rows: [
        ["r0_dense", "Naive RAG", 79.2, "#ff8a8a"],
        ["r1_hybrid", "Hybrid + RRF", 80.6, "#ffb05c"],
        ["r2_rerank", "Rerank", 84.6, "#e8b82e"],
        ["r3_hyde", "HyDE", 69.5, "#5aa9f5"],
        ["r4_multiquery", "Multi-Query", 82.7, "#8d86ff"],
        ["r5_contextual", "Contextual", 89.5, "#31b98f"],
        ["r6_raptor", "RAPTOR", 86.0, "#f083b6"],
        ["r7_crag", "CRAG", 81.1, "#2fb4c4"],
        ["r8_agentic", "Agentic", 76.5, "#8b6fe8"],
        ["adaptive", "Adaptive", 81.1, "#45435c"],
      ],
    },
    masked: {
      title: "masked 30 · 去标题 + 干扰",
      note: "去掉标题线索并加入 200 篇同前缀干扰文档；整体综合分下降约 20–24 分。",
      rows: [
        ["r0_dense", "Naive RAG", 56.2, "#ff8a8a"],
        ["r1_hybrid", "Hybrid + RRF", 60.3, "#ffb05c"],
        ["r2_rerank", "Rerank", 62.6, "#e8b82e"],
        ["r3_hyde", "HyDE", 51.4, "#5aa9f5"],
        ["r4_multiquery", "Multi-Query", 59.8, "#8d86ff"],
        ["r5_contextual", "Contextual", 65.7, "#31b98f"],
        ["r6_raptor", "RAPTOR", 58.6, "#f083b6"],
        ["r7_crag", "CRAG", 59.2, "#2fb4c4"],
        ["r8_agentic", "Agentic", 57.0, "#8b6fe8"],
        ["adaptive", "Adaptive", 59.3, "#45435c"],
      ],
    },
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const panel = $("#rung-panel");
  const tabs = $$(".rung-tab");
  const copyButtons = $$("[data-copy]");
  const copyStatus = $("#copy-status");

  function centerTab(tab, instant = false) {
    if (!tab) return;
    const bar = tab.parentElement;
    if (!bar || bar.scrollWidth <= bar.clientWidth) return;
    const barRect = bar.getBoundingClientRect();
    const tabRect = tab.getBoundingClientRect();
    const left = bar.scrollLeft + tabRect.left - barRect.left - (barRect.width - tabRect.width) / 2;
    if (instant || reduceMotion) {
      bar.scrollLeft = Math.max(0, left);
      return;
    }
    bar.scrollTo({ left: Math.max(0, left), behavior: "smooth" });
  }

  function updateRung(id) {
    const rung = rungs[id];
    if (!rung || !panel) return;

    panel.style.setProperty("--accent", rung.accent);
    panel.style.setProperty("--soft", rung.soft);

    const index = $(".panel-index", panel);
    const title = $("h3", panel);
    const sub = $(".panel-sub", panel);
    const score = $(".panel-score strong", panel);
    const plan = $(".panel-plan", panel);
    const change = $(".panel-change", panel);
    const metrics = $(".panel-metrics", panel);
    const paper = $("#panel-paper", panel);

    if (index) index.textContent = `${rung.index} · ${rung.name.toUpperCase()}`;
    if (title) title.textContent = rung.name;
    if (sub) sub.textContent = rung.sub;
    if (score) score.textContent = rung.score.toFixed(1);
    if (plan) {
      plan.innerHTML = rung.plan
        .map((step) => `<span class="plan-step">${step}</span>`)
        .join('<span class="plan-arrow" aria-hidden="true">→</span>');
    }
    if (change) {
      change.innerHTML = `<strong>只改了什么</strong> ${rung.change}`;
    }
    if (metrics) {
      metrics.innerHTML = [
        ["可答 acc", rung.metrics.acc],
        ["evidence R@1", rung.metrics.evidence],
        ["MRR@10", rung.metrics.mrr],
        ["cold $/q", rung.metrics.cost],
      ]
        .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
        .join("");
    }
    if (paper) {
      paper.href = rung.paperUrl;
      paper.innerHTML = `${rung.paper} <span aria-hidden="true">↗</span>`;
    }

    for (const tab of tabs) {
      const active = tab.dataset.rung === id;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }

    centerTab(tabs.find((tab) => tab.dataset.rung === id));
  }

  function selectRung(id, { jump = false } = {}) {
    updateRung(id);
    if (jump) {
      const target = $("#ladder");
      if (target) target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
    }
  }

  for (const tab of tabs) {
    tab.addEventListener("click", () => updateRung(tab.dataset.rung));
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      let next = null;
      if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        next = tabs[(index + 1) % tabs.length];
      } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        next = tabs[(index - 1 + tabs.length) % tabs.length];
      } else if (event.key === "Home") {
        next = tabs[0];
      } else if (event.key === "End") {
        next = tabs[tabs.length - 1];
      }
      if (next) {
        event.preventDefault();
        next.focus();
        updateRung(next.dataset.rung);
      }
    });
  }

  for (const link of $$("[data-rung-jump]")) {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      selectRung(link.dataset.rungJump, { jump: true });
    });
  }

  requestAnimationFrame(() => centerTab(document.querySelector(".rung-tab.is-active"), true));

  const graphSelect = $("#graphRung");
  const graphCaption = $("#graphCaption");
  const flow = $("#flow");

  function highlightGraph(id) {
    if (!flow) return;
    const rung = rungs[id];
    if (!rung) return;
    const plan = id === "adaptive" ? [] : rung.plan;
    flow.classList.toggle("has-focus", plan.length > 0);
    for (const node of $$(".stage-node", flow)) {
      node.classList.toggle("is-route", plan.includes(node.dataset.stage));
    }
    if (graphCaption) {
      if (id === "adaptive") {
        graphCaption.textContent = "Adaptive Router：按题型/长度选择一档，再执行对应计划。";
      } else {
        graphCaption.innerHTML =
          `${rung.name}：` +
          plan.map((step) => `<code>${step}</code>`).join(" → ") +
          "。虚线回环代表 14 个 stage 共享的返回边。";
      }
    }
  }

  if (graphSelect) {
    graphSelect.addEventListener("change", () => highlightGraph(graphSelect.value));
    highlightGraph(graphSelect.value);
  }

  const barList = $("#barList");
  const resultTitle = $("#resultsTitle");
  const resultsNote = $("#resultsNote");
  const resultTabs = $$("[data-result-tab]");

  function renderResults(name) {
    const set = resultSets[name];
    if (!set || !barList) return;
    resultTitle.textContent = set.title;
    resultsNote.textContent = set.note;
    barList.innerHTML = set.rows
      .map(
        ([, label, value, accent]) => `
          <div class="bar-row" style="--accent:${accent}">
            <div class="bar-meta"><span class="bar-dot"></span><span>${label}</span></div>
            <div class="bar-track"><div class="bar-fill" data-width="${value}"></div></div>
            <div class="bar-value">${value.toFixed(1)}</div>
          </div>`,
      )
      .join("");

    requestAnimationFrame(() => {
      for (const fill of $$(".bar-fill", barList)) {
        fill.style.width = `${fill.dataset.width}%`;
      }
    });

    for (const tab of resultTabs) {
      const active = tab.dataset.resultTab === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    }
  }

  for (const tab of resultTabs) {
    tab.addEventListener("click", () => renderResults(tab.dataset.resultTab));
  }

  renderResults("original");

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (_) {
        /* fall through */
      }
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (_) {
      ok = false;
    }
    area.remove();
    return ok;
  }

  for (const button of copyButtons) {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copy);
      if (!target) return;
      const ok = await copyText(target.textContent.trim());
      const label = $(".copy-label", button);
      if (label) label.textContent = ok ? "已复制" : "复制失败";
      button.dataset.copied = String(ok);
      if (copyStatus) copyStatus.textContent = ok ? "已复制到剪贴板" : "复制失败";
      window.setTimeout(() => {
        if (label) label.textContent = "复制";
        button.dataset.copied = "false";
      }, 1600);
    });
  }

  const progress = $("#scrollProgress");
  const header = $(".site-header");

  function onScroll() {
    if (header) header.classList.toggle("is-scrolled", window.scrollY > 8);
    if (progress) {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      progress.style.width = `${max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0}%`;
    }
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const reveals = $$(".reveal");
  if (!reduceMotion && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px" },
    );
    for (const item of reveals) observer.observe(item);
  } else {
    for (const item of reveals) item.classList.add("is-visible");
  }

  const year = $("#year");
  if (year) year.textContent = new Date().getFullYear();
})();
