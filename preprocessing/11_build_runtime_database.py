"""Build the consolidated read-only SQLite database used in production.

The article search database is copied into a new runtime database, then the
citation FTS index and incoming-link scores are added as separate logical
tables. The original build artifacts are left untouched.

Run from the localhistory repository root::

    python preprocessing/11_build_runtime_database.py
    python preprocessing/11_build_runtime_database.py --overwrite

The resulting SQLite file is suitable for a read-only API volume. Embeddings
remain separate NumPy memory maps because the search code accesses them with
``mmap_mode='r'``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
ARTICLE_DATABASE = DATA_DIR / "local_history_search.sqlite"
CITATION_DATABASE = DATA_DIR / "local_history_citations.sqlite"
LINK_COUNTS = DATA_DIR / "article_link_counts.parquet"
RUNTIME_DATABASE = DATA_DIR / "local_history_runtime.sqlite"
RUNTIME_MANIFEST = DATA_DIR / "runtime_manifest.json"


def load_importance(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["page_id", "incoming_link_count"],
    )
    frame["page_id"] = pd.to_numeric(frame["page_id"], errors="coerce")
    frame["incoming_link_count"] = pd.to_numeric(
        frame["incoming_link_count"], errors="coerce"
    ).fillna(0).clip(lower=0)
    frame = frame.dropna(subset=["page_id"]).drop_duplicates("page_id")

    log_counts = np.log1p(frame["incoming_link_count"].to_numpy(dtype="float64"))
    cap = float(np.quantile(log_counts, 0.99)) if len(log_counts) else 0.0
    frame["editorial_importance"] = (
        np.clip(log_counts / cap, 0.0, 1.0) if cap > 0 else 0.0
    )
    frame["page_id"] = frame["page_id"].astype("int64")
    frame["incoming_link_count"] = frame["incoming_link_count"].astype("int64")
    return frame[["page_id", "incoming_link_count", "editorial_importance"]]


def build_runtime_database(*, overwrite: bool) -> None:
    for path in (ARTICLE_DATABASE, CITATION_DATABASE, LINK_COUNTS):
        if not path.exists():
            raise FileNotFoundError(f"Missing required build artifact: {path}")
    if RUNTIME_DATABASE.exists() and not overwrite:
        raise FileExistsError(
            f"{RUNTIME_DATABASE} already exists; pass --overwrite to replace it"
        )

    temporary = RUNTIME_DATABASE.with_suffix(RUNTIME_DATABASE.suffix + ".building")
    if temporary.exists():
        temporary.unlink()

    print(f"Copying article database: {ARTICLE_DATABASE}")
    source = sqlite3.connect(
        f"file:{ARTICLE_DATABASE}?mode=ro&immutable=1",
        uri=True,
    )
    destination = sqlite3.connect(temporary)
    try:
        source.backup(destination, pages=10_000)
        destination.execute("PRAGMA journal_mode=OFF")
        destination.execute("PRAGMA synchronous=OFF")
        destination.execute("PRAGMA temp_store=MEMORY")

        print("Copying citation FTS index...")
        destination.execute(
            """
            CREATE VIRTUAL TABLE citation_fts USING fts5(
                page_id UNINDEXED,
                sentence,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        destination.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        destination.execute(
            f"ATTACH DATABASE ? AS citation_source",
            (str(CITATION_DATABASE),),
        )
        destination.execute(
            "INSERT INTO citation_fts(page_id, sentence) "
            "SELECT page_id, sentence FROM citation_source.citation_fts"
        )
        destination.execute(
            "INSERT INTO metadata(key, value) "
            "SELECT key, value FROM citation_source.metadata"
        )
        destination.commit()
        destination.execute("DETACH DATABASE citation_source")
        destination.execute("INSERT INTO citation_fts(citation_fts) VALUES('optimize')")

        print("Adding article importance scores...")
        importance = load_importance(LINK_COUNTS)
        destination.execute(
            """
            CREATE TABLE article_importance (
                page_id INTEGER PRIMARY KEY,
                incoming_link_count INTEGER NOT NULL,
                editorial_importance REAL NOT NULL
            )
            """
        )
        destination.executemany(
            "INSERT INTO article_importance "
            "(page_id, incoming_link_count, editorial_importance) VALUES (?, ?, ?)",
            importance.itertuples(index=False, name=None),
        )
        destination.execute(
            "CREATE INDEX article_importance_score_idx "
            "ON article_importance(editorial_importance)"
        )

        destination.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("runtime_database_version", "1"),
        )
        destination.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("link_count_rows", str(len(importance))),
        )
        destination.commit()
        integrity = destination.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    except BaseException:
        destination.close()
        source.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        destination.close()
        source.close()

    temporary.replace(RUNTIME_DATABASE)
    with sqlite3.connect(
        f"file:{RUNTIME_DATABASE}?mode=ro&immutable=1", uri=True
    ) as runtime_connection:
        article_count = runtime_connection.execute(
            "SELECT count(*) FROM articles"
        ).fetchone()[0]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": RUNTIME_DATABASE.name,
        "database_size_bytes": RUNTIME_DATABASE.stat().st_size,
        "article_database": ARTICLE_DATABASE.name,
        "citation_database": CITATION_DATABASE.name,
        "link_counts": LINK_COUNTS.name,
        "article_count": article_count,
        "importance_rows": len(importance),
        "separate_memory_mapped_files": [
            "kalm_embeddings.npy",
            "kalm_embedding_page_ids.npy",
            "city_query_embeddings.npy",
        ],
    }
    RUNTIME_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {RUNTIME_DATABASE}")
    print(f"Manifest: {RUNTIME_MANIFEST}")
    print(f"Database size: {RUNTIME_DATABASE.stat().st_size / (1024 ** 3):.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_runtime_database(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
