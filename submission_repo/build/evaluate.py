"""Run a question set against the indexed RAG bot and save answers/sources.

Input JSON may be a list of strings or objects containing ``question`` and
optional ``answer``/``expected`` fields. Example:

    python build/evaluate.py my_questions.json --out data/my_run.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.answer import answer_from_chunks, retrieve


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"\w+", (value or "").casefold()))


def run(question_records: list, output_path: Path) -> list[dict]:
    console = Console()
    table = Table(title="RAG question run", show_lines=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Question", max_width=42)
    table.add_column("Answer", max_width=58)
    table.add_column("Sources", max_width=42)
    results = []

    for index, record in enumerate(question_records, 1):
        if isinstance(record, str):
            question, expected = record, ""
        else:
            question = str(record.get("question", "")).strip()
            expected = str(record.get("expected", record.get("answer", ""))).strip()
        if not question:
            continue
        hits = retrieve(question)
        answer = answer_from_chunks(question, hits)
        source_urls = list(dict.fromkeys(
            hit.get("metadata", {}).get("url", "") for hit in hits
            if hit.get("metadata", {}).get("url")
        ))
        exact_or_contains = None
        if expected:
            expected_norm, answer_norm = _normalized(expected), _normalized(answer)
            exact_or_contains = bool(expected_norm and expected_norm in answer_norm)
        result = {
            "question": question,
            "expected": expected or None,
            "answer": answer,
            "expected_found_in_answer": exact_or_contains,
            "sources": source_urls,
            "evidence": [
                {
                    "title": hit.get("metadata", {}).get("title", ""),
                    "section": hit.get("metadata", {}).get("section", ""),
                    "kind": hit.get("metadata", {}).get("kind", "text"),
                    "text": hit.get("text", ""),
                    "url": hit.get("metadata", {}).get("url", ""),
                }
                for hit in hits
            ],
        }
        results.append(result)
        table.add_row(str(index), question, answer, "\n".join(source_urls[:3]))
        console.print(table)
        table.rows.clear()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    console.print(f"Saved detailed answers and retrieved evidence to {output_path}")
    if any(item["expected_found_in_answer"] is not None for item in results):
        checked = [item for item in results if item["expected_found_in_answer"] is not None]
        matches = sum(item["expected_found_in_answer"] for item in checked)
        console.print(f"Expected-answer text matched in {matches}/{len(checked)} responses (rough check; review evidence manually).")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG answers and inspect their retrieved evidence.")
    parser.add_argument("questions", type=Path, help="JSON list of questions or {question, answer} objects")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "evaluation_results.json")
    args = parser.parse_args()
    records = json.loads(args.questions.read_text())
    if not isinstance(records, list):
        raise SystemExit("Question file must contain a JSON list")
    run(records, args.out)
