# Building and checking the InnoWing knowledge index

Run these commands from `submission_repo/`:

```bash
source .venv/bin/activate
python check_setup.py
python build/scrape.py
python build/index.py
python build/images.py
python main.py "What does the InnoWing offer?"
python build/inspect_retrieval.py "What does the InnoWing offer?"
python build/evaluate.py my_questions.json --out data/my_run.json
```

`build/scrape.py` displays a live terminal dashboard with the current page,
extracted text, page counts, image references, and linked PDFs. The crawler
collects public WordPress posts and pages from both supplied sites, including
titles, URLs, content types, publish/modified dates, authors, categories, tags,
and public SEO/Open Graph/schema metadata when the API exposes it. It also
collects image alt text/captions and linked PDFs, extracting PDF text and
document properties when available.

The crawler uses smaller REST pages, retries transient timeouts, and writes an
ignored checkpoint after every API page. If a run fails, rerun
`python build/scrape.py` to resume from the last completed page; use
`python build/scrape.py --fresh` to intentionally start both sites over. It
only replaces the final `data/*.json` snapshot after both sites finish, so a
failed site cannot erase the last complete snapshot.

`build/index.py` makes heading-aware overlapping chunks, adds source/date/type
metadata, and builds the persistent Chroma collection. `build/images.py` caches
vision descriptions by image URL so an interrupted run can resume; it indexes
those descriptions with the image alt text, caption, and source pages.

At question time, retrieval combines vector similarity with BM25 keyword
matching. The answer builder keeps the prompt bounded, limits repeated chunks
from the same page, and gives the model source URLs and section names. Use
`build/inspect_retrieval.py` to inspect the exact context before tuning an
answer; add `--answer` to call the chat deployment too. Use `build/evaluate.py`
with your question JSON to save model answers, expected answers, source URLs,
and retrieved evidence in one report.

The API key stays in `.env` and must not be committed. `check_setup.py` should
show both the chat and embedding calls as `PASS` before the index can be built.
