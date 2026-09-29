"""Collect public InnoWing and InnoAcademy content for the RAG index.

Run from any working directory with ``python build/scrape.py``. An
interactive terminal gets a Rich dashboard showing the latest extracted
page and text; redirected output gets page-by-page progress lines.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_PATH = DATA_DIR / "crawl_checkpoint.json"
USER_AGENT = "InnoWingChallengeRAG/1.0 (public website content indexing)"
PER_PAGE = 25
REQUEST_PAUSE = 0.12
REQUEST_TIMEOUT = (10, 60)
REQUEST_ATTEMPTS = 4

# Both supplied sites expose their public WordPress posts and pages at the
# host root. Crawling the full InnoWing host is necessary because its posts
# are not permalinked below the supplied /innowing1/ landing-page path.
SITES = [
    {
        "name": "InnoWing",
        "start_url": "https://innowings.engg.hku.hk/innowing1/",
        "scope_prefix": "/",
    },
    {
        "name": "InnoAcademy",
        "start_url": "https://innoacademy.engg.hku.hk/",
        "scope_prefix": "/",
    },
]


def _normalized_url(url: str) -> str:
    parts = urlsplit(url)
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _same_host(url: str, start_url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    site_host = (urlsplit(start_url).hostname or "").lower().removeprefix("www.")
    return bool(host) and host == site_host


def in_scope(url: str, site: dict) -> bool:
    if not _same_host(url, site["start_url"]):
        return False
    path = urlsplit(url).path or "/"
    prefix = site["scope_prefix"]
    if prefix == "/":
        return not any(path.startswith(x) for x in ("/wp-admin/", "/wp-json/"))
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _html_text(markup: str) -> str:
    """Extract readable text while preserving paragraph and heading breaks."""
    if not markup:
        return ""
    soup = BeautifulSoup(markup, "html.parser")
    for node in soup.select("script, style, noscript, iframe, svg, form, button"):
        node.decompose()
    for node in soup.select("h1, h2, h3, h4, h5, h6"):
        level = int(node.name[1])
        node.insert_before(f"\n{'#' * level} ")
        node.insert_after("\n")
    for node in soup.select("br, p, li, blockquote, tr"):
        node.insert_before("\n")
        node.insert_after("\n")
    raw = soup.get_text(" ", strip=False).replace("\xa0", " ")
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return "\n".join(lines)


def _html_metadata(markup: str) -> dict:
    """Read metadata tags and JSON-LD when the supplied markup contains them."""
    if not markup:
        return {}
    soup = BeautifulSoup(markup, "html.parser")
    tags = {}
    for tag in soup.select("meta[name], meta[property], meta[itemprop]"):
        key = (tag.get("name") or tag.get("property") or tag.get("itemprop") or "").strip().lower()
        value = re.sub(r"\s+", " ", tag.get("content", "")).strip()
        if key and value and key not in tags:
            tags[key] = value
    canonical = soup.select_one('link[rel~="canonical"][href]')

    schema_types, schema_names = [], []

    def visit(value, budget):
        if budget[0] <= 0:
            return
        budget[0] -= 1
        if isinstance(value, list):
            for child in value:
                visit(child, budget)
        elif isinstance(value, dict):
            types = value.get("@type", [])
            if isinstance(types, str):
                types = [types]
            for item in types:
                if isinstance(item, str) and item not in schema_types:
                    schema_types.append(item)
            name = value.get("name") or value.get("alternateName")
            if isinstance(name, str) and name.strip() and name.strip() not in schema_names:
                schema_names.append(name.strip())
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child, budget)

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            visit(json.loads(script.string or script.get_text()), [3000])
        except (json.JSONDecodeError, TypeError):
            continue

    result = {
        "meta_description": tags.get("description") or tags.get("og:description") or tags.get("twitter:description", ""),
        "og_title": tags.get("og:title") or tags.get("twitter:title", ""),
        "og_description": tags.get("og:description", ""),
        "og_image": tags.get("og:image") or tags.get("twitter:image", ""),
        "og_type": tags.get("og:type", ""),
        "og_site_name": tags.get("og:site_name", ""),
        "twitter_card": tags.get("twitter:card", ""),
        "canonical_url": _normalized_url(canonical.get("href")) if canonical and canonical.get("href") else "",
        "schema_types": schema_types,
        "schema_names": schema_names[:20],
    }
    return {key: value for key, value in result.items() if value}


def _srcset_candidates(value: str) -> list[tuple[int, str]]:
    candidates = []
    for item in (value or "").split(","):
        bits = item.strip().split()
        if not bits:
            continue
        width = 0
        if len(bits) > 1:
            match = re.match(r"(\d+)w", bits[1])
            density = re.match(r"([\d.]+)x", bits[1])
            width = int(match.group(1)) if match else int(float(density.group(1)) * 1000) if density else 0
        candidates.append((width, bits[0]))
    return candidates


def _image_from_markup(img, page_url: str) -> dict | None:
    candidates = []
    for attr in ("src", "data-src", "data-lazy-src", "data-original"):
        value = img.get(attr)
        if value and not value.startswith("data:"):
            candidates.append((0, value))
    for attr in ("srcset", "data-srcset"):
        candidates.extend(_srcset_candidates(img.get(attr, "")))
    picture = img.find_parent("picture")
    for source in picture.select("source[srcset]") if picture else []:
        candidates.extend(_srcset_candidates(source.get("srcset", "")))
    if not candidates:
        return None
    src = urljoin(page_url, max(candidates, key=lambda pair: pair[0])[1])
    if not urlsplit(src).scheme.startswith("http"):
        return None
    figure = img.find_parent("figure")
    caption = _html_text(str(figure.find("figcaption"))) if figure and figure.find("figcaption") else ""
    return {
        "src": _normalized_url(src),
        "alt": re.sub(r"\s+", " ", img.get("alt", "")).strip(),
        "caption": caption,
        "page": page_url,
    }


def _dedupe_values(*values: str) -> str:
    unique = []
    for value in values:
        value = re.sub(r"\s+", " ", value or "").strip()
        if value and value not in unique:
            unique.append(value)
    return " | ".join(unique)


def extract(html: str, url: str) -> dict:
    """Extract readable page text, images, and linked PDF files from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = _html_text(str(soup.title)) if soup.title else ""
    body = soup.select_one("main, article, #content, .site-main") or soup.body or soup
    text = _html_text(str(body))
    images = []
    for img in body.select("img"):
        image = _image_from_markup(img, url)
        if image:
            images.append(image)

    pdfs = []
    for anchor in body.select("a[href]"):
        href = urljoin(url, anchor.get("href", ""))
        if urlsplit(href).path.lower().endswith(".pdf"):
            pdfs.append({
                "url": _normalized_url(href),
                "title": _html_text(str(anchor)) or Path(urlsplit(href).path).name,
                "page": url,
            })
    return {
        "url": _normalized_url(url),
        "title": title,
        "text": text,
        "images": images,
        "pdfs": pdfs,
        "metadata": _html_metadata(html),
    }


def _embedded_images(item: dict, page_url: str) -> list[dict]:
    result = []
    for media in item.get("_embedded", {}).get("wp:featuredmedia", []):
        media_url = media.get("source_url")
        original = media.get("media_details", {}).get("sizes", {}).get("full", {}).get("source_url")
        media_url = original or media_url
        if media_url:
            result.append({
                "src": _normalized_url(media_url),
                "alt": _html_text(media.get("alt_text", "")),
                "caption": _html_text(media.get("caption", {}).get("rendered", "")),
                "page": page_url,
            })
    return result


def _terms(item: dict) -> list[str]:
    terms = []
    for group in item.get("_embedded", {}).get("wp:term", []):
        for term in group:
            taxonomy = term.get("taxonomy", "")
            if taxonomy and taxonomy not in {"category", "post_tag"}:
                continue
            name = re.sub(r"\s+", " ", term.get("name", "")).strip()
            if name and name not in terms:
                terms.append(name)
    return terms


def _author_name(item: dict) -> str:
    authors = item.get("_embedded", {}).get("author", [])
    return re.sub(r"\s+", " ", authors[0].get("name", "")).strip() if authors else ""


def _rest_metadata(item: dict, extracted: dict) -> dict:
    """Normalize core WordPress and common SEO-plugin metadata."""
    groups = {"category": [], "post_tag": []}
    for group in item.get("_embedded", {}).get("wp:term", []):
        for term in group:
            taxonomy = term.get("taxonomy", "")
            if taxonomy in groups:
                name = re.sub(r"\s+", " ", term.get("name", "")).strip()
                if name and name not in groups[taxonomy]:
                    groups[taxonomy].append(name)

    yoast = item.get("yoast_head_json") or {}
    rank_math = item.get("rank_math") or item.get("rank_math_seo") or {}
    if not isinstance(yoast, dict):
        yoast = {}
    if not isinstance(rank_math, dict):
        rank_math = {}
    html_meta = extracted.get("metadata", {})
    yoast_images = yoast.get("og_image") or []
    og_image = (yoast_images[0].get("url", "") if yoast_images and isinstance(yoast_images[0], dict) else "")
    metadata = {
        "author": _author_name(item),
        "categories": groups["category"],
        "tags": groups["post_tag"],
        "meta_description": yoast.get("description") or rank_math.get("description") or html_meta.get("meta_description", ""),
        "og_title": yoast.get("og_title") or rank_math.get("facebook_title") or html_meta.get("og_title", ""),
        "og_description": yoast.get("og_description") or rank_math.get("facebook_description") or html_meta.get("og_description", ""),
        "og_image": og_image or html_meta.get("og_image", ""),
        "og_type": yoast.get("og_type") or html_meta.get("og_type", ""),
        "og_site_name": yoast.get("og_site_name") or html_meta.get("og_site_name", ""),
        "twitter_card": yoast.get("twitter_card") or html_meta.get("twitter_card", ""),
        "canonical_url": yoast.get("canonical") or rank_math.get("canonical_url") or html_meta.get("canonical_url", ""),
        "schema_types": html_meta.get("schema_types", []),
        "schema_names": html_meta.get("schema_names", []),
    }
    wp_meta = item.get("meta") or {}
    if isinstance(wp_meta, dict):
        footnotes = wp_meta.get("footnotes")
        if isinstance(footnotes, str) and footnotes.strip():
            metadata["footnotes"] = _html_text(footnotes)
        for key, value in wp_meta.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if not any(token in normalized_key for token in ("seo", "description", "canonical", "open_graph", "og_", "schema")):
                continue
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            metadata[normalized_key] = _html_text(str(value))[:2000]
    return {key: value for key, value in metadata.items() if value}


def _page_from_rest(item: dict, kind: str, site: dict) -> dict:
    url = _normalized_url(item.get("link", ""))
    title = _html_text(item.get("title", {}).get("rendered", ""))
    extracted = extract(item.get("content", {}).get("rendered", ""), url)
    unique_images = {}
    for image in extracted["images"] + _embedded_images(item, url):
        src = image["src"]
        if src not in unique_images:
            unique_images[src] = image
        else:
            old = unique_images[src]
            old["alt"] = _dedupe_values(old.get("alt"), image.get("alt"))
            old["caption"] = _dedupe_values(old.get("caption"), image.get("caption"))

    terms = _terms(item)
    text = extracted["text"] or _html_text(item.get("excerpt", {}).get("rendered", ""))
    metadata = _rest_metadata(item, extracted)
    descriptors = []
    if terms:
        descriptors.append("Topics: " + ", ".join(terms))
    if metadata.get("author"):
        descriptors.append("Author: " + metadata["author"])
    if metadata.get("meta_description"):
        descriptors.append("Page description: " + metadata["meta_description"])
    if metadata.get("og_title") and metadata["og_title"] != title:
        descriptors.append("Open Graph title: " + metadata["og_title"])
    if metadata.get("og_description") and metadata["og_description"] != metadata.get("meta_description"):
        descriptors.append("Open Graph description: " + metadata["og_description"])
    if metadata.get("schema_types"):
        descriptors.append("Structured content type: " + ", ".join(metadata["schema_types"]))
    if metadata.get("schema_names"):
        descriptors.append("Structured names: " + ", ".join(metadata["schema_names"]))
    if metadata.get("footnotes"):
        descriptors.append("Footnotes: " + metadata["footnotes"])
    if descriptors:
        text = "\n".join(descriptors + ([text] if text else []))
    date = item.get("date", "")
    year = date[:4] if re.match(r"^\d{4}", date) else ""
    source_host = (urlsplit(site["start_url"]).hostname or site["name"].lower()).lower()
    return {
        "record_id": f"{source_host}:{kind}:{item.get('id', '')}",
        "url": url,
        "title": title,
        "text": text,
        "site": site["name"],
        "page_type": kind,
        "date": date,
        "year": year,
        "modified": item.get("modified", ""),
        "metadata": metadata,
        "images": list(unique_images.values()),
        "pdfs": extracted["pdfs"],
    }


class CrawlDashboard:
    """Live terminal dashboard with crawl status and the latest extracted text."""

    def __init__(self, enabled: bool = True):
        self.console = Console()
        self.enabled = enabled and self.console.is_terminal
        self.live = None
        self.site_name = "Starting"
        self.site_seen = 0
        self.site_total = 0
        self.site_docs = 0
        self.site_images = 0
        self.site_pdfs = 0
        self.total_docs = 0
        self.current = None
        self.recent = deque(maxlen=4)

    def start(self):
        if self.enabled:
            self.live = Live(self._render(), console=self.console, refresh_per_second=8, transient=False)
            self.live.start()
        else:
            self.console.print("[bold cyan]InnoWing + InnoAcademy crawler[/bold cyan]")

    def stop(self):
        if self.live:
            self.live.stop()

    def _render(self):
        status = Table.grid(expand=True, padding=(0, 1))
        status.add_column(style="bold cyan", no_wrap=True)
        status.add_column()
        status.add_column(style="bold cyan", no_wrap=True)
        status.add_column()
        progress = f"{self.site_seen:,} / {self.site_total:,}" if self.site_total else f"{self.site_seen:,}"
        status.add_row("Site", self.site_name, "REST records", progress)
        status.add_row("Pages kept", str(self.site_docs), "Image refs", str(self.site_images))
        status.add_row("PDF links", str(self.site_pdfs), "Total pages kept", str(self.total_docs))
        status_panel = Panel(status, title="[bold white]LIVE CRAWL[/bold white]", border_style="cyan", box=box.ROUNDED)

        if self.current:
            body = Group(
                Text(self.current.get("title") or "Untitled page", style="bold bright_cyan"),
                Text(self.current.get("url", ""), style="dim cyan"),
                Text(""),
                Text(self.current.get("text", "")[:720] or "(No readable body text)", overflow="fold"),
            )
            current_panel = Panel(body, title="Latest extracted text", border_style="green", box=box.ROUNDED)
        else:
            current_panel = Panel(Text("Connecting to the WordPress content API…", style="dim"),
                                  title="Latest extracted text", border_style="green", box=box.ROUNDED)

        recent = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold magenta")
        recent.add_column("Recent pages", ratio=2)
        recent.add_column("Text preview", ratio=3)
        for title, preview in self.recent:
            recent.add_row(Text(title[:64], overflow="ellipsis"), Text(preview[:100], overflow="ellipsis", style="dim"))
        recent_panel = Panel(recent, title="Recent crawl", border_style="magenta", box=box.ROUNDED)
        return Group(status_panel, current_panel, recent_panel)

    def _update(self, force: bool = False):
        if self.live and (force or self.site_seen % 8 == 0):
            self.live.update(self._render(), refresh=True)

    def begin_site(self, name: str):
        self.site_name = name
        self.site_seen = self.site_total = self.site_docs = self.site_images = self.site_pdfs = 0
        self.current = None
        self.recent.clear()
        self._update(force=True)
        if not self.live:
            self.console.print(f"\n[bold cyan]── {name} ──[/bold cyan]")

    def expected(self, total: int):
        self.site_total += total
        self._update(force=True)

    def record(self, page: dict | None):
        self.site_seen += 1
        if page:
            self.site_docs += 1
            self.total_docs += 1
            self.site_images += len(page["images"])
            self.site_pdfs += len(page["pdfs"])
            self.current = page
            preview = re.sub(r"\s+", " ", page.get("text", "")).strip()
            self.recent.append((page.get("title") or page["url"], preview))
            if not self.live:
                self.console.print(f"[green]PAGE[/green] {page.get('title') or '(untitled)'}")
                self.console.print(f"[dim]{page['url']}[/dim]")
                self.console.print(Text(preview[:360] or "(No readable text)", overflow="fold"))
        self._update()


def _wp_api_root(site: dict) -> str:
    host = urlsplit(site["start_url"])
    return f"{host.scheme}://{host.netloc}/wp-json/wp/v2"


def _get_json(session: requests.Session, url: str, params: dict) -> tuple[list, int]:
    transient_statuses = {429, 500, 502, 503, 504}
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code in transient_statuses and attempt < REQUEST_ATTEMPTS:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(float(retry_after), 30)
                except ValueError:
                    delay = min(2 ** (attempt - 1), 8)
                response.close()
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"Expected a WordPress content list from {response.url}")
            return payload, int(response.headers.get("X-WP-Total", len(payload)))
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt >= REQUEST_ATTEMPTS:
                raise requests.ConnectionError(
                    f"{type(exc).__name__} after {attempt} attempts for {url} (page={params.get('page')}, per_page={params.get('per_page')})"
                ) from exc
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Could not fetch {url}")


def _new_site_checkpoint() -> dict:
    return {
        "pages": [],
        "images": [],
        "pdfs": [],
        "totals": {},
        "seen": {},
        "next_kind": "posts",
        "next_page": 1,
        "complete": False,
    }


def _save_checkpoint(checkpoint: dict) -> None:
    _write_json_atomic(CHECKPOINT_PATH, checkpoint)


def crawl_site(site: dict, dashboard: CrawlDashboard, checkpoint: dict, root_checkpoint: dict) -> tuple[list[dict], list[dict], list[dict]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    api_root = _wp_api_root(site)
    pages = checkpoint.setdefault("pages", [])
    page_ids = {item.get("record_id") or item.get("url") for item in pages}
    images_by_url = {item["src"]: item for item in checkpoint.setdefault("images", [])}
    pdfs_by_url = {item["url"]: item for item in checkpoint.setdefault("pdfs", [])}
    totals = checkpoint.setdefault("totals", {})
    seen = checkpoint.setdefault("seen", {})
    kinds = ("posts", "pages")
    next_kind = checkpoint.get("next_kind", "posts")
    start_kind_index = kinds.index(next_kind) if next_kind in kinds else len(kinds)

    dashboard.site_seen = sum(seen.values())
    dashboard.site_docs = len(pages)
    dashboard.site_images = sum(len(page.get("images", [])) for page in pages)
    dashboard.site_pdfs = sum(len(page.get("pdfs", [])) for page in pages)
    for total in totals.values():
        dashboard.expected(total)
    dashboard._update(force=True)

    try:
        for kind_index, kind in enumerate(kinds):
            if kind_index < start_kind_index:
                continue
            page_number = int(checkpoint.get("next_page", 1)) if kind_index == start_kind_index else 1
            while True:
                endpoint = f"{api_root}/{kind}"
                params = {"per_page": PER_PAGE, "page": page_number, "_embed": "1", "orderby": "date", "order": "asc"}
                records, total = _get_json(session, endpoint, params)
                if kind not in totals:
                    totals[kind] = total
                    dashboard.expected(total)
                else:
                    totals[kind] = total
                if not records:
                    if (page_number - 1) * PER_PAGE < total:
                        raise RuntimeError(f"WordPress returned no records for {site['name']} {kind} page {page_number} of {total}")
                    if kind_index + 1 < len(kinds):
                        checkpoint["next_kind"] = kinds[kind_index + 1]
                        checkpoint["next_page"] = 1
                    else:
                        checkpoint["next_kind"] = None
                        checkpoint["complete"] = True
                    checkpoint["pages"] = pages
                    checkpoint["images"] = list(images_by_url.values())
                    checkpoint["pdfs"] = list(pdfs_by_url.values())
                    _save_checkpoint(root_checkpoint)
                    break
                if len(records) < PER_PAGE and page_number * PER_PAGE < total:
                    raise RuntimeError(
                        f"Incomplete WordPress page for {site['name']} {kind} page {page_number}: "
                        f"received {len(records)} of {PER_PAGE} while {total} records remain"
                    )

                for item in records:
                    url = _normalized_url(item.get("link", ""))
                    if not in_scope(url, site):
                        dashboard.record(None)
                        continue
                    page = _page_from_rest(item, "post" if kind == "posts" else "page", site)
                    if not page["text"].strip() and not page["images"]:
                        dashboard.record(None)
                        continue
                    page_id = page.get("record_id") or page["url"]
                    if page_id not in page_ids:
                        pages.append(page)
                        page_ids.add(page_id)
                    for image in page["images"]:
                        src = image["src"]
                        if src not in images_by_url:
                            images_by_url[src] = {**image, "pages": [page["url"]], "site": site["name"]}
                        else:
                            existing = images_by_url[src]
                            existing["alt"] = _dedupe_values(existing.get("alt"), image.get("alt"))
                            existing["caption"] = _dedupe_values(existing.get("caption"), image.get("caption"))
                            existing["site"] = _dedupe_values(existing.get("site"), site["name"])
                            if page["url"] not in existing["pages"]:
                                existing["pages"].append(page["url"])
                    for pdf in page["pdfs"]:
                        if pdf["url"] not in pdfs_by_url:
                            pdfs_by_url[pdf["url"]] = {**pdf, "pages": [pdf["page"]], "site": site["name"]}
                        else:
                            existing = pdfs_by_url[pdf["url"]]
                            if pdf["page"] not in existing["pages"]:
                                existing["pages"].append(pdf["page"])
                    dashboard.record(page)

                seen[kind] = seen.get(kind, 0) + len(records)
                if len(records) < PER_PAGE or page_number * PER_PAGE >= total:
                    if kind_index + 1 < len(kinds):
                        checkpoint["next_kind"] = kinds[kind_index + 1]
                        checkpoint["next_page"] = 1
                    else:
                        checkpoint["next_kind"] = None
                        checkpoint["complete"] = True
                else:
                    checkpoint["next_kind"] = kind
                    checkpoint["next_page"] = page_number + 1
                checkpoint["pages"] = pages
                checkpoint["images"] = list(images_by_url.values())
                checkpoint["pdfs"] = list(pdfs_by_url.values())
                _save_checkpoint(root_checkpoint)
                if checkpoint.get("next_kind") != kind:
                    break
                page_number += 1
                time.sleep(REQUEST_PAUSE)
    finally:
        session.close()
    return pages, list(images_by_url.values()), list(pdfs_by_url.values())


def extract_pdf_text(pdf_links: list[dict]) -> list[dict]:
    """Extract text from linked PDFs and retain page numbers and source URLs."""
    if not pdf_links:
        return []
    try:
        from pypdf import PdfReader
    except ImportError:
        Console(stderr=True).print("[yellow]PDF extraction skipped. Install pypdf to index linked PDFs.[/yellow]")
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    documents = []
    for index, pdf in enumerate(pdf_links, 1):
        try:
            response = session.get(pdf["url"], timeout=(10, 60), stream=True)
            response.raise_for_status()
            content = bytearray()
            for block in response.iter_content(64 * 1024):
                if block:
                    content.extend(block)
                    if len(content) > 30 * 1024 * 1024:
                        raise ValueError("PDF exceeds 30 MB limit")
            reader = PdfReader(io.BytesIO(content), strict=False)
            pdf_meta = reader.metadata or {}
            pdf_title = str(getattr(pdf_meta, "title", "") or "").strip()
            pdf_author = str(getattr(pdf_meta, "author", "") or "").strip()
            pdf_subject = str(getattr(pdf_meta, "subject", "") or "").strip()
            raw_keywords = pdf_meta.get("/Keywords", "") if hasattr(pdf_meta, "get") else ""
            if isinstance(raw_keywords, list):
                keywords = [str(value).strip() for value in raw_keywords if str(value).strip()]
            else:
                keywords = [value.strip() for value in re.split(r"[,;]", str(raw_keywords or "")) if value.strip()]
            created = str(pdf_meta.get("/CreationDate", "") if hasattr(pdf_meta, "get") else "")
            year_match = re.search(r"(?:D:)?(\d{4})", created)
            extracted_pages = []
            for page_no, pdf_page in enumerate(reader.pages, 1):
                text = re.sub(r"\s+", " ", pdf_page.extract_text() or "").strip()
                if text:
                    extracted_pages.append(f"[PDF page {page_no}]\n{text}")
            if extracted_pages:
                enriched_text = []
                if pdf_subject:
                    enriched_text.append("PDF subject: " + pdf_subject)
                if keywords:
                    enriched_text.append("PDF keywords: " + ", ".join(keywords))
                enriched_text.append("\n\n".join(extracted_pages))
                documents.append({
                    "record_id": "pdf:" + hashlib.sha256(pdf["url"].encode("utf-8")).hexdigest()[:20],
                    "url": pdf["url"],
                    "title": pdf_title or pdf.get("title") or Path(urlsplit(pdf["url"]).path).name,
                    "text": "\n\n".join(enriched_text),
                    "site": pdf.get("site", ""),
                    "page_type": "pdf",
                    "date": "",
                    "year": year_match.group(1) if year_match else "",
                    "related_pages": pdf.get("pages", []),
                    "pdf_page_count": len(reader.pages),
                    "metadata": {
                        key: value for key, value in {
                            "author": pdf_author,
                            "subject": pdf_subject,
                            "keywords": keywords,
                            "creator": str(pdf_meta.get("/Creator", "") if hasattr(pdf_meta, "get") else "").strip(),
                            "producer": str(pdf_meta.get("/Producer", "") if hasattr(pdf_meta, "get") else "").strip(),
                        }.items() if value
                    },
                })
        except Exception as exc:
            Console(stderr=True).print(f"[yellow]PDF {index}/{len(pdf_links)} skipped ({type(exc).__name__}):[/yellow] {pdf['url']}")
        if index < len(pdf_links):
            time.sleep(REQUEST_PAUSE)
    session.close()
    return documents


def _empty_checkpoint() -> dict:
    return {"version": 1, "per_page": PER_PAGE, "sites": {}}


def _load_checkpoint(fresh: bool) -> tuple[dict, bool]:
    if fresh or not CHECKPOINT_PATH.exists():
        return _empty_checkpoint(), False
    try:
        checkpoint = json.loads(CHECKPOINT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return _empty_checkpoint(), False
    compatible = (
        checkpoint.get("version") == 1
        and checkpoint.get("per_page") == PER_PAGE
        and isinstance(checkpoint.get("sites"), dict)
    )
    if not compatible:
        return _empty_checkpoint(), False
    if all(checkpoint["sites"].get(site["name"], {}).get("complete") for site in SITES):
        return _empty_checkpoint(), False
    return checkpoint, bool(checkpoint["sites"])


def _combined_snapshot(checkpoint: dict) -> tuple[list[dict], list[dict], list[dict]]:
    pages, page_ids = [], set()
    images_by_url, pdfs_by_url = {}, {}
    for site in SITES:
        state = checkpoint["sites"][site["name"]]
        for page in state.get("pages", []):
            page_id = f"{page.get('site', site['name'])}:{page.get('record_id') or page.get('url')}"
            if page_id not in page_ids:
                pages.append(page)
                page_ids.add(page_id)
        for image in state.get("images", []):
            existing = images_by_url.get(image["src"])
            if existing is None:
                images_by_url[image["src"]] = image
            else:
                existing["alt"] = _dedupe_values(existing.get("alt"), image.get("alt"))
                existing["caption"] = _dedupe_values(existing.get("caption"), image.get("caption"))
                existing["site"] = _dedupe_values(existing.get("site"), image.get("site"))
                existing["pages"] = list(dict.fromkeys(existing.get("pages", []) + image.get("pages", [])))
        for pdf in state.get("pdfs", []):
            existing = pdfs_by_url.get(pdf["url"])
            if existing is None:
                pdfs_by_url[pdf["url"]] = pdf
            else:
                existing["pages"] = list(dict.fromkeys(existing.get("pages", []) + pdf.get("pages", [])))
                existing["site"] = _dedupe_values(existing.get("site"), pdf.get("site"))
    return pages, list(images_by_url.values()), list(pdfs_by_url.values())


def run_crawl(live: bool = True, fresh: bool = False) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    checkpoint, resuming = _load_checkpoint(fresh=fresh)
    for site in SITES:
        checkpoint["sites"].setdefault(site["name"], _new_site_checkpoint())
    if resuming:
        Console().print("[cyan]Resuming from the last saved page checkpoint.[/cyan]")

    dashboard = CrawlDashboard(enabled=live)
    dashboard.total_docs = sum(len(state.get("pages", [])) for state in checkpoint["sites"].values())
    dashboard.start()
    _save_checkpoint(checkpoint)
    try:
        for site in SITES:
            site_checkpoint = checkpoint["sites"][site["name"]]
            if site_checkpoint.get("complete"):
                continue
            dashboard.begin_site(site["name"])
            crawl_site(site, dashboard, site_checkpoint, checkpoint)
            _save_checkpoint(checkpoint)
    finally:
        dashboard.stop()

    if not all(checkpoint["sites"][site["name"]].get("complete") for site in SITES):
        raise RuntimeError("Crawl stopped before both sites completed; rerun to resume from the checkpoint")

    pages, images, pdfs = _combined_snapshot(checkpoint)
    pdf_documents = extract_pdf_text(pdfs)
    _save_crawl_files(pages, images, pdfs, pdf_documents)
    Console().print(
        f"\n[bold green]Crawl saved:[/bold green] {len(pages):,} pages · "
        f"{len(images):,} unique images · {len(pdfs):,} linked PDFs · "
        f"{len(pdf_documents):,} PDFs with extractable text"
    )
    return pages, images, pdfs, pdf_documents


def _write_json_atomic(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2))
    temporary.replace(path)


def _save_crawl_files(pages: list[dict], images: list[dict], pdfs: list[dict], pdf_documents: list[dict]) -> None:
    outputs = {
        "pages.json": pages,
        "images.json": images,
        "pdfs.json": pdfs,
        "pdf_documents.json": pdf_documents,
    }
    for name, content in outputs.items():
        _write_json_atomic(DATA_DIR / name, content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl the supplied InnoWing and InnoAcademy website scopes.")
    parser.add_argument("--plain", action="store_true", help="print extracted text rather than using the live dashboard")
    parser.add_argument("--fresh", action="store_true", help="discard an interrupted checkpoint and start both sites over")
    args = parser.parse_args()
    try:
        run_crawl(live=not args.plain, fresh=args.fresh)
    except KeyboardInterrupt:
        Console().print("\n[yellow]Crawl interrupted. Progress is checkpointed; rerun to resume.[/yellow]")
        raise SystemExit(130)
    except Exception as exc:
        Console(stderr=True).print(
            f"[bold red]Crawl failed:[/bold red] {type(exc).__name__}: {exc}\n"
            "Progress is checkpointed. Rerun the same command to resume; use --fresh to start over."
        )
        raise SystemExit(1)
