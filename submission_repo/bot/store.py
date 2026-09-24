"""Vector store wrapper. GIVEN. You should not need to edit this file.

A thin layer over chromadb, kept deliberately small so you can see
through it. If you need something it does not do, edit it or bypass it.
"""
from pathlib import Path
from collections import Counter
import math
import re

import chromadb

from bot.llm import embed

DEFAULT_PATH = str(Path(__file__).resolve().parents[1] / "data" / "chroma")
DEFAULT_NAME = "chatbot"

# One PersistentClient per path, cached for the life of the process.
# chromadb caches internal state per path, so deleting the folder and
# opening a fresh PersistentClient while an earlier one is still alive
# in this process corrupts the connection: the next call fails with
# "attempt to write a readonly database" or "database is locked". This
# only bites if get_store(reset=True) runs more than once in one Python
# process, e.g. build/index.py and bot/answer.py both imported into the
# same interactive session, so it will not show up in a normal
# `python build/index.py` run, only in a notebook or a test harness.
_stores: dict = {}


def get_store(path: str = DEFAULT_PATH, name: str = DEFAULT_NAME, reset: bool = False):
    """Open (or create) a persistent Chroma collection.

    reset=True deletes and recreates the collection, not the directory,
    so it is safe to call more than once in the same process.
    """
    if path not in _stores:
        # First time this path is opened in this process. Safe to wipe a
        # stale, wrong-chromadb-version index here, since no client for
        # this path exists yet.
        if reset and Path(path).exists():
            import shutil
            shutil.rmtree(path)
        try:
            _stores[path] = chromadb.PersistentClient(path=path)
        except KeyError as e:
            raise RuntimeError(
                f"chromadb could not read the index at {path} ({e}). It was likely "
                f"built by a different chromadb version. Delete that folder and "
                f"rebuild, or install the pinned version from requirements.txt."
            ) from None

    client = _stores[path]
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name)


def add_to_store(store, texts: list[str], metadatas: list[dict],
                 ids: list[str] = None, batch_size: int = 128) -> None:
    """Embed and upsert. Stable IDs make repeated indexing idempotent."""
    if len(texts) != len(metadatas):
        raise ValueError("texts and metadatas must have the same length")
    if not texts:
        return
    ids = ids or [f"c{i}" for i in range(len(texts))]
    if len(ids) != len(texts):
        raise ValueError("ids and texts must have the same length")
    for i in range(0, len(texts), batch_size):
        sl = slice(i, i + batch_size)
        store.upsert(
            ids=ids[sl], documents=texts[sl],
            embeddings=embed(texts[sl]), metadatas=metadatas[sl],
        )


def query(store, question: str, k: int = 5, where: dict = None) -> list[dict]:
    """Return the k nearest chunks as dicts with text, metadata, distance.

    Chroma returns squared L2 distance by default, so lower is closer.
    """
    count = store.count()
    if count == 0:
        return []
    r = store.query(query_embeddings=embed([question]), n_results=min(max(1, k), count), where=where or None)
    return [
        {"id": item_id, "text": d, "metadata": m, "distance": dist}
        for item_id, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with",
}


def _tokens(text: str) -> list[str]:
    return [word.casefold() for word in _WORD.findall(text or "") if len(word) > 1 and word.casefold() not in _STOP_WORDS]


def _bm25_scores(question: str, documents: list[str]) -> list[float]:
    """Small in-process BM25 pass to complement embeddings for names/numbers."""
    query_terms = _tokens(question)
    tokenized = [_tokens(document) for document in documents]
    if not query_terms or not tokenized:
        return [0.0] * len(documents)
    frequencies = [Counter(tokens) for tokens in tokenized]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(1, len(lengths))
    document_frequency = Counter()
    for counter in frequencies:
        document_frequency.update(counter.keys())
    total = len(documents)
    scores = [0.0] * total
    k1, b = 1.5, 0.75
    for term in set(query_terms):
        df = document_frequency.get(term, 0)
        if not df:
            continue
        idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
        for index, counter in enumerate(frequencies):
            tf = counter.get(term, 0)
            if tf:
                denominator = tf + k1 * (1 - b + b * lengths[index] / max(1, average_length))
                scores[index] += idf * tf * (k1 + 1) / denominator
    return scores


def hybrid_query(store, question: str, k: int = 8, where: dict = None) -> list[dict]:
    """Fuse vector and BM25 rankings so semantic matches and exact terms count."""
    count = store.count()
    if count == 0:
        return []
    candidate_count = min(count, max(32, k * 4))
    vector_results = query(store, question, k=candidate_count, where=where)
    get_args = {"include": ["documents", "metadatas"]}
    if where:
        get_args["where"] = where
    corpus = store.get(**get_args)
    ids = corpus.get("ids", [])
    documents = corpus.get("documents", [])
    metadatas = corpus.get("metadatas", [])
    scores = _bm25_scores(question, documents)
    lexical_order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

    fused = {}
    details = {}
    for rank, result in enumerate(vector_results, 1):
        item_id = result["id"]
        fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (60 + rank)
        details[item_id] = result
    for rank, index in enumerate(lexical_order[:candidate_count], 1):
        if scores[index] <= 0:
            break
        item_id = ids[index]
        fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (60 + rank)
        details.setdefault(item_id, {
            "id": item_id,
            "text": documents[index],
            "metadata": metadatas[index],
            "distance": None,
        })
    ranked = sorted(fused, key=fused.__getitem__, reverse=True)
    results = []
    for item_id in ranked[:k]:
        item = details[item_id]
        item["hybrid_score"] = fused[item_id]
        results.append(item)
    return results
