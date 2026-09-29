"""Build a section-aware, metadata-rich Chroma index from the crawl files."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.store import add_to_store, get_store

DATA_DIR = ROOT / "data"
CHUNK_SIZE = 950
OVERLAP = 120


def _split_long(text: str, limit: int, overlap: int) -> list[str]:
    """Prefer sentence boundaries, falling back to overlapping characters."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                parts.append(current)
                current = ""
            step = max(1, limit - overlap)
            parts.extend(sentence[i:i + limit] for i in range(0, len(sentence), step))
        elif not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= limit:
            current += " " + sentence
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict[str, str]]:
    """Group paragraphs by heading and keep the section heading with each chunk."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be between 0 and size")

    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_paragraphs: list[str] = []

    def flush_section():
        nonlocal current_paragraphs
        if current_paragraphs:
            sections.append((current_heading, current_paragraphs))
            current_paragraphs = []

    for line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        heading_match = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading_match:
            flush_section()
            current_heading = heading_match.group(1).strip()
        else:
            current_paragraphs.append(line)
    flush_section()

    result = []
    for heading, paragraphs in sections:
        heading_prefix = f"{heading}\n" if heading else ""
        content_limit = max(1, size - len(heading_prefix))
        pieces = []
        buffer = ""
        for paragraph in paragraphs:
            for part in _split_long(paragraph, content_limit, overlap):
                if not buffer:
                    buffer = part
                elif len(buffer) + len(part) + 2 <= content_limit:
                    buffer += "\n\n" + part
                else:
                    pieces.append(buffer)
                    # Keep a small tail from the previous paragraph so a fact
                    # near a boundary retains its local context.
                    tail = buffer[-overlap:].strip() if overlap else ""
                    buffer = (tail + "\n\n" + part).strip() if tail and len(tail) + len(part) + 2 <= content_limit else part
        if buffer:
            pieces.append(buffer)
        for piece in pieces:
            result.append({"section": heading, "text": (heading_prefix + piece).strip()})
    return result


def _stable_id(url: str, record_id: str, position: int, piece: str, kind: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{record_id}\0{url}\0{position}\0{piece}".encode("utf-8")).hexdigest()[:32]
    return f"{kind}_{digest}"


def _metadata(record: dict, position: int, section: str, kind: str = "text") -> dict:
    record_id = record.get("record_id") or f"{record.get('site', '')}:{record.get('url', '')}:{record.get('title', '')}"
    metadata = {
        "record_id": str(record_id)[:500],
        "url": str(record.get("url", "")),
        "title": str(record.get("title", ""))[:500],
        "site": str(record.get("site", "")),
        "page_type": str(record.get("page_type", "page")),
        "date": str(record.get("date", "")),
        "year": str(record.get("year", "")),
        "section": str(section)[:300],
        "position": int(position),
        "kind": kind,
    }
    related = record.get("related_pages")
    if related:
        metadata["related_pages"] = " | ".join(str(url) for url in related)[:2000]
    source_metadata = record.get("metadata", {})
    if isinstance(source_metadata, dict):
        for key in (
            "author", "categories", "tags", "meta_description", "og_title",
            "og_description", "og_image", "og_type", "og_site_name",
            "twitter_card", "canonical_url", "schema_types", "schema_names",
            "subject", "keywords", "creator", "producer", "footnotes",
        ):
            value = source_metadata.get(key)
            if isinstance(value, list):
                value = " | ".join(str(item) for item in value if item)
            if value:
                metadata[key] = str(value)[:2000]
    if record.get("pdf_page_count"):
        metadata["pdf_page_count"] = int(record["pdf_page_count"])
    return metadata


def build_index(pages: list[dict], reset: bool = True, documents: list[dict] | None = None):
    texts, metadatas, ids = [], [], []
    records = list(pages) + list(documents or [])
    for record in records:
        if not record.get("text", "").strip():
            continue
        kind = "pdf" if record.get("page_type") == "pdf" else "text"
        pieces = chunk(record["text"])
        for position, piece in enumerate(pieces):
            # The page title is prepended so a standalone section chunk keeps
            # its topic even when retrieval returns it without neighbors.
            document_text = "\n".join(filter(None, [record.get("title", ""), piece["text"]])).strip()
            if not document_text:
                continue
            texts.append(document_text)
            metadatas.append(_metadata(record, position, piece["section"], kind=kind))
            record_id = record.get("record_id") or f"{record.get('site', '')}:{record.get('url', '')}:{record.get('title', '')}"
            ids.append(_stable_id(record.get("url", ""), record_id, position, document_text, kind))

    store = get_store(reset=reset)
    if texts:
        add_to_store(store, texts, metadatas, ids=ids)
    print(f"indexed {len(texts):,} chunks from {len(records):,} source records")
    return store


def load_crawl_data():
    pages_path = DATA_DIR / "pages.json"
    if not pages_path.exists():
        raise FileNotFoundError(f"{pages_path} is missing; run build/scrape.py first")
    pages = json.loads(pages_path.read_text())
    pdf_path = DATA_DIR / "pdf_documents.json"
    documents = json.loads(pdf_path.read_text()) if pdf_path.exists() else []
    return pages, documents


if __name__ == "__main__":
    pages, documents = load_crawl_data()
    build_index(pages, reset=True, documents=documents)
