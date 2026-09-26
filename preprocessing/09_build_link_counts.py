"""Build incoming Wikipedia-link counts for Local History articles.

This is a one-time preprocessing job. It counts links from all normal
Wikipedia pages to normal Wikipedia pages, then keeps only targets that are
already present in the Local History article index.

The three Wikimedia tables are joined as follows::

    page.page_id                  source page ID
    pagelinks.pl_from             -> page.page_id
    pagelinks.pl_target_id        -> linktarget.lt_id
    linktarget.(namespace,title) -> page.(namespace,title)

Run from the repository root::

    python preprocessing/09_build_link_counts.py
    python preprocessing/09_build_link_counts.py --overwrite

The output is intentionally small and suitable for joining into the runtime
ranking data later. Raw Wikimedia dumps are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def resolve_data_dir() -> Path:
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data",
        Path(__file__).resolve().parent.parent / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()
PAGE_SQL = DATA_DIR / "enwiki-latest-page.sql.gz"
LINKTARGET_SQL = DATA_DIR / "enwiki-latest-linktarget.sql.gz"
PAGELINKS_SQL = DATA_DIR / "enwiki-latest-pagelinks.sql.gz"
ARTICLE_DATABASE = DATA_DIR / "local_history_search.sqlite"
OUTPUT_PATH = DATA_DIR / "article_link_counts.parquet"
METADATA_PATH = DATA_DIR / "article_link_counts.json"

# The dump rows contain SQL-escaped strings. These expressions deliberately
# extract only the columns needed by this job.
PAGE_RE = re.compile(rb"\((\d+),0,'((?:[^'\\]|''|\\.)*)'")
LINKTARGET_RE = re.compile(rb"\((\d+),0,'((?:[^'\\]|''|\\.)*)'")
PAGELINK_RE = re.compile(rb"\((\d+),(\d+),(\d+)\)")


def unescape_sql_string(value: str) -> str:
    """Decode the common SQL escapes used in Wikimedia dumps."""
    return value.replace("\\'", "'").replace("''", "'").replace("\\\\", "\\")


def normalized_title(value: str) -> str:
    """Match the normalized title form used by MediaWiki link targets."""
    return value.strip().replace(" ", "_")


def stream_matches(path: Path, table: str, pattern: re.Pattern[bytes]):
    """Yield regex matches from a decompressed dump without huge allocations.

    Some current Wikimedia dumps contain one very large INSERT statement on a
    single line. Chunking the decompressed stream keeps memory bounded while
    retaining enough overlap for tuples split across chunk boundaries.
    """
    needle = f"INSERT INTO `{table}` VALUES".encode()
    chunk_size = 8 * 1024 * 1024
    overlap = 4096
    started = False
    buffer = b""
    # The current linktarget dump has trailing bytes after its gzip member.
    # The system gzip reader can decode the valid member and ignore those
    # bytes; Python's gzip module raises on them.
    process = subprocess.Popen(
        ["gzip", "-cd", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    try:
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            if not started:
                marker = buffer.find(needle)
                if marker < 0:
                    buffer = buffer[-len(needle):]
                    continue
                buffer = buffer[marker + len(needle):]
                started = True

            safe_length = max(0, len(buffer) - overlap)
            for match in pattern.finditer(buffer[:safe_length]):
                yield match
            buffer = buffer[safe_length:]

        if started:
            for match in pattern.finditer(buffer):
                yield match
    finally:
        process.stdout.close()
        return_code = process.wait()
        # -13 is SIGPIPE when a caller intentionally stops consuming the
        # generator early (for example, during a smoke test).
        if return_code not in (0, 2, -13):
            raise RuntimeError(f"gzip failed for {path} with exit code {return_code}")


def load_local_articles() -> dict[str, int]:
    """Return normalized article title -> Local History page ID."""
    if not ARTICLE_DATABASE.exists():
        raise FileNotFoundError(
            f"Missing {ARTICLE_DATABASE}. Build the SQLite search database first."
        )

    uri = f"file:{ARTICLE_DATABASE}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("SELECT page_id, title FROM articles")
        titles = {}
        for page_id, title in rows:
            if title is None:
                continue
            titles[normalized_title(str(title))] = int(page_id)

    print(f"Loaded {len(titles):,} Local History article titles")
    return titles


def load_local_page_ids(local_titles: dict[str, int]) -> dict[str, int]:
    """Find Local History page IDs in the current Wikipedia page dump."""
    page_ids: dict[str, int] = {}
    print(f"Pass 1/3: resolving Local History titles in {PAGE_SQL.name}")
    for row_number, match in enumerate(stream_matches(PAGE_SQL, "page", PAGE_RE), start=1):
        page_id = int(match.group(1))
        title = normalized_title(unescape_sql_string(match.group(2).decode()))
        if title in local_titles:
            page_ids[title] = page_id
        if row_number % 1_000_000 == 0:
            print(f"  scanned {row_number:,} page rows | found {len(page_ids):,}", end="\r")
    print(f"\n  Found {len(page_ids):,} matching Wikipedia pages")
    return page_ids


def load_target_ids(local_page_ids: dict[str, int]) -> dict[int, int]:
    """Resolve linktarget IDs that point to Local History articles."""
    target_ids: dict[int, int] = {}
    print(f"Pass 2/3: resolving article link targets in {LINKTARGET_SQL.name}")
    for row_number, match in enumerate(stream_matches(LINKTARGET_SQL, "linktarget", LINKTARGET_RE), start=1):
        target_id = int(match.group(1))
        title = normalized_title(unescape_sql_string(match.group(2).decode()))
        page_id = local_page_ids.get(title)
        if page_id is not None:
            target_ids[target_id] = page_id
        if row_number % 1_000_000 == 0:
            print(f"  scanned {row_number:,} linktarget rows | matched {len(target_ids):,}", end="\r")
    print(f"\n  Matched {len(target_ids):,} article targets")
    return target_ids


def count_incoming_links(target_ids: dict[int, int]) -> Counter[int]:
    """Count normal-namespace source pages for each Local History target."""
    counts: Counter[int] = Counter()
    rows_seen = 0
    print(f"Pass 3/3: counting incoming links in {PAGELINKS_SQL.name}")
    for row_number, match in enumerate(stream_matches(PAGELINKS_SQL, "pagelinks", PAGELINK_RE), start=1):
        source_id = int(match.group(1))
        source_namespace = int(match.group(2))
        target_id = int(match.group(3))
        if source_namespace != 0:
            continue
        target_page_id = target_ids.get(target_id)
        if target_page_id is None or source_id == target_page_id:
            continue
        # pagelinks has a primary key on (pl_from, pl_target_id), so each
        # accepted row is already a distinct source link for this target.
        counts[target_page_id] += 1
        rows_seen += 1
        if row_number % 1_000_000 == 0:
            print(f"  scanned {row_number:,} pagelink rows | counted {rows_seen:,} links", end="\r")
    print(f"\n  Counted {rows_seen:,} article-to-article links")
    return counts


def save_counts(local_titles: dict[str, int], local_page_ids: dict[str, int], counts: Counter[int], *, overwrite: bool) -> None:
    if OUTPUT_PATH.exists() and not overwrite:
        raise FileExistsError(f"{OUTPUT_PATH} already exists; pass --overwrite to replace it")

    page_to_title = {page_id: title for title, page_id in local_page_ids.items()}
    rows = [
        {
            "page_id": page_id,
            "title": page_to_title.get(page_id),
            "incoming_link_count": int(counts.get(page_id, 0)),
        }
        for page_id in local_titles.values()
    ]
    frame = pd.DataFrame(rows).sort_values("incoming_link_count", ascending=False)
    frame.to_parquet(OUTPUT_PATH, index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_namespace": 0,
        "target_namespace": 0,
        "self_links_excluded": True,
        "local_history_articles": len(local_titles),
        "resolved_local_history_pages": len(local_page_ids),
        "targets_with_incoming_links": int((frame["incoming_link_count"] > 0).sum()),
        "total_incoming_links": int(frame["incoming_link_count"].sum()),
        "sources": {
            "page": PAGE_SQL.name,
            "linktarget": LINKTARGET_SQL.name,
            "pagelinks": PAGELINKS_SQL.name,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(frame):,} rows to {OUTPUT_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (PAGE_SQL, LINKTARGET_SQL, PAGELINKS_SQL):
        if not path.exists():
            raise FileNotFoundError(f"Missing Wikimedia dump: {path}")

    local_titles = load_local_articles()
    local_page_ids = load_local_page_ids(local_titles)
    target_ids = load_target_ids(local_page_ids)
    counts = count_incoming_links(target_ids)
    save_counts(local_titles, local_page_ids, counts, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
