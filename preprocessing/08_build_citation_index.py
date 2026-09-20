"""Build a sentence-level SQLite FTS5 index for citation-like passages.

Only sentences that satisfy the search ranker's citation heuristic are stored.
At query time, FTS5 can therefore find citation-context city mentions without
reading or splitting the full text of every candidate article.

Run from the localhistory directory:

    python preprocessing/08_build_citation_index.py
    python preprocessing/08_build_citation_index.py --overwrite
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import os
from pathlib import Path
import sqlite3
import sys
import time


try:
    from localhistory.citation_index import (
        CITATION_HEURISTIC_VERSION,
        iter_citation_sentences,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from citation_index import (  # type: ignore[no-redef]
        CITATION_HEURISTIC_VERSION,
        iter_citation_sentences,
    )


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
ARTICLE_DATABASE_PATH = DATA_DIR / "local_history_search.sqlite"
CITATION_DATABASE_PATH = DATA_DIR / "local_history_citations.sqlite"

SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE citation_fts USING fts5(
    page_id UNINDEXED,
    sentence,
    tokenize='unicode61 remove_diacritics 2'
);
"""


def extract_context_rows(
    rows: list[tuple[int, str | None]],
) -> list[tuple[int, str]]:
    """Extract index rows from a source batch in a worker process."""
    contexts = []
    for page_id, full_text in rows:
        contexts.extend(
            (page_id, sentence)
            for sentence in iter_citation_sentences(full_text)
        )
    return contexts


def build_database(
    source: Path,
    destination: Path,
    *,
    batch_size: int,
    workers: int,
    overwrite: bool,
) -> None:
    if not source.exists():
        raise FileNotFoundError(
            f"Article search database not found: {source}\n"
            "Build it with preprocessing/05_build_sqlite_search.py first."
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Citation database already exists: {destination}\n"
            "Pass --overwrite to rebuild it."
        )

    temporary = destination.with_suffix(destination.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_uri = f"file:{source}?mode=ro&immutable=1"
    source_connection = sqlite3.connect(source_uri, uri=True)
    output_connection = sqlite3.connect(temporary)
    started = time.perf_counter()

    try:
        article_count = source_connection.execute(
            "SELECT count(*) FROM articles WHERE NOT is_redirect"
        ).fetchone()[0]
        print(f"Source: {source}")
        print(f"Articles: {article_count:,}")
        print(f"Building: {temporary}")

        output_connection.execute("PRAGMA journal_mode=OFF")
        output_connection.execute("PRAGMA synchronous=OFF")
        output_connection.execute("PRAGMA temp_store=MEMORY")
        output_connection.execute("PRAGMA cache_size=-200000")
        output_connection.executescript(SCHEMA)

        cursor = source_connection.execute(
            """
            SELECT page_id, full_text
            FROM articles
            WHERE NOT is_redirect
            ORDER BY page_id
            """
        )

        def store_batch(contexts, article_batch_size):
            nonlocal processed, context_count
            if contexts:
                output_connection.executemany(
                    "INSERT INTO citation_fts(page_id, sentence) VALUES (?, ?)",
                    contexts,
                )
                context_count += len(contexts)
            output_connection.commit()
            processed += article_batch_size
            elapsed = time.perf_counter() - started
            print(
                f"  indexed {processed:,}/{article_count:,} articles, "
                f"{context_count:,} contexts ({elapsed:.1f}s)"
            )

        processed = 0
        context_count = 0
        batches = iter(lambda: cursor.fetchmany(batch_size), [])
        if workers <= 1:
            for rows in batches:
                store_batch(extract_context_rows(rows), len(rows))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                while group := list(islice(batches, workers * 2)):
                    contexts = executor.map(extract_context_rows, group)
                    for rows, context_rows in zip(group, contexts):
                        store_batch(context_rows, len(rows))

        print("Optimizing citation FTS index...")
        output_connection.execute(
            "INSERT INTO citation_fts(citation_fts) VALUES('optimize')"
        )
        output_connection.execute(
            "INSERT INTO citation_fts(citation_fts) VALUES('integrity-check')"
        )
        metadata = {
            "citation_heuristic_version": CITATION_HEURISTIC_VERSION,
            "source_path": str(source),
            "source_size": str(source.stat().st_size),
            "source_mtime_ns": str(source.stat().st_mtime_ns),
            "article_count": str(article_count),
            "context_count": str(context_count),
        }
        output_connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        output_connection.commit()

        indexed_count = output_connection.execute(
            "SELECT count(*) FROM citation_fts"
        ).fetchone()[0]
        if indexed_count != context_count:
            raise RuntimeError(
                f"Expected {context_count:,} contexts, found {indexed_count:,}"
            )
        integrity = output_connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    except BaseException:
        source_connection.close()
        output_connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        source_connection.close()
        output_connection.close()

    temporary.replace(destination)
    elapsed = time.perf_counter() - started
    size_mib = destination.stat().st_size / (1024 ** 2)
    print(f"Complete: {destination}")
    print(f"Citation contexts: {context_count:,}")
    print(f"Database size: {size_mib:.1f} MiB")
    print(f"Elapsed: {elapsed:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ARTICLE_DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=CITATION_DATABASE_PATH)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel extraction processes (default: up to 4)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_database(
        args.source.expanduser().resolve(),
        args.output.expanduser().resolve(),
        batch_size=args.batch_size,
        workers=max(1, args.workers),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
