"""Inspect the exact evidence selected for a RAG question.

Example: python build/inspect_retrieval.py "What courses use the Innovation Wing?"
Add --answer to also call the configured chat deployment.
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.answer import build_context, rag_answer, retrieve


def main():
    parser = argparse.ArgumentParser(description="Inspect hybrid vector/BM25 retrieval and bounded answer context.")
    parser.add_argument("question", nargs="+", help="question to retrieve evidence for")
    parser.add_argument("--answer", action="store_true", help="also generate an answer with the chat model")
    args = parser.parse_args()
    question = " ".join(args.question)
    console = Console()
    hits = retrieve(question)
    if not hits:
        console.print("[yellow]No retrieved evidence. Build data/chroma first.[/yellow]")
        return

    console.print(Panel(Text(question, style="bold"), title="Question", border_style="cyan"))
    for rank, hit in enumerate(hits, 1):
        metadata = hit.get("metadata", {})
        title = metadata.get("title") or "Untitled source"
        section = metadata.get("section") or "Overview"
        url = metadata.get("url", "")
        distance = hit.get("distance")
        score = hit.get("hybrid_score")
        subtitle = f"{metadata.get('site', '')} · {metadata.get('kind', 'text')} · hybrid {score:.4f}"
        if distance is not None:
            subtitle += f" · vector distance {distance:.3f}"
        console.print(Panel(hit["text"], title=f"{rank}. {title} — {section}", subtitle=f"{subtitle}\n{url}", border_style="green"))

    console.print(Panel(build_context(hits), title="Bounded context sent to the answer model", border_style="magenta"))
    if args.answer:
        console.print(Panel(rag_answer(question), title="Answer", border_style="yellow"))


if __name__ == "__main__":
    main()
