"""Describe collected images once, then add searchable descriptions to Chroma."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.llm import describe_image
from bot.store import add_to_store, get_store

DATA_DIR = ROOT / "data"
CACHE_PATH = DATA_DIR / "descriptions.json"
USER_AGENT = "InnoWingChallengeRAG/1.0 (public website image descriptions)"

DESCRIPTION_PROMPT = """Describe this website image as evidence for an assistant answering factual questions about the InnoWing and InnoAcademy.

Capture: the overall scene or subject; every clearly visible object and a careful count when the count is unambiguous; named equipment, rooms, signs, labels, diagrams, and their spatial relationships; visible text as accurately as possible; and useful distinguishing colors, materials, or layout. For diagrams, explain the labels and connections. Keep exact names and numbers. Separate what is visibly certain from what is unclear. Do not guess identities, measurements, or facts that cannot be read or seen. Write a concise but information-rich description in complete sentences."""


def _image_id(src: str) -> str:
    return "img_" + hashlib.sha256(src.encode("utf-8")).hexdigest()[:32]


def _image_prompt(image: dict) -> str:
    hints = []
    if image.get("alt"):
        hints.append(f"Website alt text: {image['alt']}")
    if image.get("caption"):
        hints.append(f"Website caption: {image['caption']}")
    if hints:
        return DESCRIPTION_PROMPT + "\n\nMetadata hints (verify them against the image):\n" + "\n".join(hints)
    return DESCRIPTION_PROMPT


def describe_all(images: list[dict]) -> dict:
    """Vision describe each unique image, caching completed work by image URL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    missing = [image for image in images if image.get("src") and not cache.get(image["src"])]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=Console(), transient=False,
    ) as progress:
        task = progress.add_task("Image descriptions", total=len(missing))
        for image in images:
            src = image.get("src", "")
            if not src or cache.get(src):
                continue
            try:
                parsed = urlsplit(src)
                suffix = Path(parsed.path).suffix.lower()
                if suffix == ".svg":
                    raise ValueError("SVG images are not supported by the vision endpoint")
                response = session.get(src, timeout=(10, 45), stream=True)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if not content_type.startswith("image/"):
                    raise ValueError(f"URL returned {content_type or 'non-image content'}")
                content = bytearray()
                for block in response.iter_content(64 * 1024):
                    if block:
                        content.extend(block)
                        if len(content) > 15 * 1024 * 1024:
                            raise ValueError("image exceeds 15 MB limit")
                if not suffix:
                    suffix = mimetypes.guess_extension(content_type) or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
                    temp.write(content)
                    temp.flush()
                    cache[src] = describe_image(temp.name, _image_prompt(image)).strip()
                # Persist after every successful image so an interrupted run
                # can continue without paying to describe completed images.
                CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
            except Exception as exc:
                print(f"image skipped ({type(exc).__name__}): {src}")
            progress.advance(task)
            time.sleep(0.08)

    session.close()
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    Console().print(f"[green]{len(cache):,} image descriptions cached[/green]")
    return cache


def index_descriptions(descriptions: dict, images: list[dict]) -> None:
    """Index visual descriptions, captions, and alt text with stable IDs."""
    by_src = {image["src"]: image for image in images if image.get("src")}
    texts, metadatas, ids = [], [], []
    for src, image in by_src.items():
        description = descriptions.get(src, "")
        context = "\n".join(filter(None, [
            image.get("alt", ""),
            image.get("caption", ""),
            description,
            "Referenced from: " + ", ".join(image.get("pages", [])) if image.get("pages") else "",
        ])).strip()
        if not context:
            continue
        texts.append(context[:8000])
        metadatas.append({
            "url": (image.get("pages") or [src])[0],
            "title": (image.get("alt") or image.get("caption") or "Image")[:500],
            "site": str(image.get("site", "")),
            "page_type": "image",
            "year": "",
            "date": "",
            "section": "Visual description",
            "position": 0,
            "kind": "image",
            "image": src,
            "pages": " | ".join(image.get("pages", []))[:2000],
        })
        ids.append(_image_id(src))
    if texts:
        add_to_store(get_store(reset=False), texts, metadatas, ids=ids)
    Console().print(f"[green]Indexed {len(texts):,} visual descriptions[/green]")


if __name__ == "__main__":
    images_path = DATA_DIR / "images.json"
    if not images_path.exists():
        raise SystemExit("data/images.json is missing; run build/scrape.py first")
    image_records = json.loads(images_path.read_text())
    image_descriptions = describe_all(image_records)
    index_descriptions(image_descriptions, image_records)
