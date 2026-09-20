"""Build the persistent SQLite FTS5 article search database.

The regular ``articles`` table stores the source rows. ``article_fts`` is an
external-content FTS5 index over title, first_paragraph, and full_text. The
unicode61 tokenizer supports word and phrase lookup without the much larger
character-level index produced by a trigram tokenizer.

Run from the localhistory directory:

    python preprocessing/05_build_sqlite_search.py
    python preprocessing/05_build_sqlite_search.py --overwrite
"""

import argparse
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
INDEX_PATH = DATA_DIR / "local_history_index.parquet"
DATABASE_PATH = DATA_DIR / "local_history_search.sqlite"

COLUMNS = [
    "page_id",
    "title",
    "first_paragraph",
    "full_text",
    "is_redirect",
    "is_list_article",
    "lat",
    "lon",
    "birth_date",
    "death_date",
    "inception",
    "dissolved",
    "start_time",
    "end_time",
    "point_in_time",
    "instance_of",
    "occupation",
    "country",
    "location",
    "wikidata_id",
    "hop",
    "canonical_date",
    "year",
    "entity_class",
]

SCHEMA = """
CREATE TABLE articles (
    page_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    first_paragraph TEXT,
    full_text TEXT,
    is_redirect INTEGER NOT NULL,
    is_list_article INTEGER NOT NULL,
    lat REAL,
    lon REAL,
    birth_date TEXT,
    death_date TEXT,
    inception TEXT,
    dissolved TEXT,
    start_time TEXT,
    end_time TEXT,
    point_in_time TEXT,
    instance_of TEXT,
    occupation TEXT,
    country TEXT,
    location TEXT,
    wikidata_id TEXT,
    hop INTEGER,
    canonical_date TEXT,
    year REAL,
    entity_class TEXT
);

CREATE VIRTUAL TABLE article_fts USING fts5(
    title,
    first_paragraph,
    full_text,
    content='articles',
    content_rowid='page_id',
    tokenize='unicode61 remove_diacritics 2'
);
"""


def sqlite_value(value):
    """Convert Arrow/pandas values into SQLite-compatible scalar values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(list(value), ensure_ascii=False)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return int(value)
    return value


def batch_rows(frame: pd.DataFrame):
    for row in frame[COLUMNS].itertuples(index=False, name=None):
        yield tuple(sqlite_value(value) for value in row)


def build_database(
    source: Path,
    destination: Path,
    *,
    batch_size: int,
    overwrite: bool,
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Parquet index not found: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Database already exists: {destination}\n"
            "Pass --overwrite to rebuild it."
        )

    temporary = destination.with_suffix(destination.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(source)
    placeholders = ", ".join("?" for _ in COLUMNS)
    columns = ", ".join(COLUMNS)
    insert_sql = f"INSERT INTO articles ({columns}) VALUES ({placeholders})"

    print(f"Source: {source}")
    print(f"Rows: {parquet.metadata.num_rows:,}")
    print(f"Building: {temporary}")

    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-200000")
        connection.executescript(SCHEMA)

        inserted = 0
        for arrow_batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=COLUMNS,
        ):
            frame = arrow_batch.to_pandas()
            connection.executemany(insert_sql, batch_rows(frame))
            connection.commit()
            inserted += len(frame)
            print(f"  imported {inserted:,}/{parquet.metadata.num_rows:,}")

        print("Building coordinate index...")
        connection.execute(
            """
            CREATE INDEX articles_lat_lon_idx ON articles(lat, lon)
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            """
        )
        connection.commit()

        print("Building unicode61 full-text index...")
        connection.execute(
            "INSERT INTO article_fts(article_fts) VALUES('rebuild')"
        )
        connection.execute(
            "INSERT INTO article_fts(article_fts) VALUES('optimize')"
        )
        connection.commit()

        article_count = connection.execute(
            "SELECT count(*) FROM articles"
        ).fetchone()[0]
        if article_count != parquet.metadata.num_rows:
            raise RuntimeError(
                f"Expected {parquet.metadata.num_rows:,} rows, found {article_count:,}"
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    except BaseException:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()

    temporary.replace(destination)
    size_gib = destination.stat().st_size / (1024 ** 3)
    print(f"Complete: {destination}")
    print(f"Database size: {size_gib:.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=INDEX_PATH)
    parser.add_argument("--output", type=Path, default=DATABASE_PATH)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_database(
        args.source.expanduser().resolve(),
        args.output.expanduser().resolve(),
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
