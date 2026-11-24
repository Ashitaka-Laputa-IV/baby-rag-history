"""CLI: ask one question or run the whole ladder."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from .config import load_settings
from .eval import evaluate, summarize, write_reports
from .graph import build_graph, run_one
from .llm import LLMClient
from .retrieval import Services
from .rungs import ALL_RUNGS, LABELS, PAPERS, RUNGS

app = typer.Typer(add_completion=False, help="Nine RAG milestones in one LangGraph.")
console = Console()


@app.command()
def info() -> None:
    table = Table(title="baby-rag-history · the ladder")
    table.add_column("rung", style="cyan")
    table.add_column("milestone")
    table.add_column("plan")
    table.add_column("anchor")
    for rung in ALL_RUNGS:
        plan = "adaptive" if rung == "adaptive" else " → ".join(RUNGS[rung])
        table.add_row(rung, LABELS.get(rung, ""), plan, PAPERS.get(rung, ""))
    console.print(table)


@app.command()
def ask(question: str, rung: str = typer.Option("r1_hybrid", help="r0..r8 or adaptive"),
        mode: str = typer.Option("ladder", help="ladder | adaptive")) -> None:
    settings = load_settings()
    services = Services(settings)
    llm = LLMClient(settings)
    graph = build_graph(services, llm)
    result = run_one(graph, question=question, rung=rung, mode=mode)
    console.print(f"\n[bold cyan]{rung}[/bold cyan] · plan: {' → '.join(result['plan'])}")
    console.print(f"\n[bold]{result['answer']}[/bold]\n")
    for i, hit in enumerate(result["hits"], start=1):
        console.print(f"[{i}] {hit['chunk_id']}  ({hit['source']}, {hit['score']:.4f})")
    console.print(f"\n[bold]${result['cost'].get('usd', 0):.5f}[/bold] · "
                  f"{result['cost'].get('latency_ms', 0)} ms · "
                  f"abstain={result['abstain']}")


@app.command("eval")
def eval_cmd(rungs: str = typer.Option("all", help="comma-separated rungs or all"),
             limit: int = typer.Option(0, help="limit questions"),
             workers: int = typer.Option(4)) -> None:
    settings = load_settings()
    chosen = ALL_RUNGS if rungs == "all" else [r.strip() for r in rungs.split(",") if r.strip()]
    rows = evaluate(settings, rungs=chosen, limit=limit, workers=workers)
    summary = summarize(rows)
    write_reports(rows, summary)
    console.print_json(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
