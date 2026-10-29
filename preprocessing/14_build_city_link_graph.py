"""Build a one-hop Wikipedia link graph for matched city articles.

The graph is kept separate from the article search database. A city seed is a
non-redirect article from ``city_article_matches.parquet``; outgoing links from
those seeds become graph-distance-one candidates.

Usage::

    uv run --extra preprocessing python preprocessing/14_build_city_link_graph.py

Use ``--city "Paris, France"`` to build a small graph for experimentation.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
import re
import sqlite3

import pandas as pd


def resolve_data_dir() -> Path:
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = [
        Path(__file__).resolve().parents[2] / "data",
        Path(__file__).resolve().parents[1] / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()
MATCHES = DATA_DIR / "city_article_matches.parquet"
PAGE_DUMP = DATA_DIR / "enwiki-latest-page.sql.gz"
LINKTARGET_DUMP = DATA_DIR / "enwiki-latest-linktarget.sql.gz"
PAGELINKS_DUMP = DATA_DIR / "enwiki-latest-pagelinks.sql.gz"
OUTPUT = DATA_DIR / "city_link_graph.sqlite"

PAGE_RE = re.compile(rb"\((\d+),(\d+),'((?:''|[^'])*)',(\d+),")
LINKTARGET_RE = re.compile(rb"\((\d+),(\-?\d+),'((?:''|[^'])*)'")
PAGELINK_RE = re.compile(rb"\((\d+),(\-?\d+),(\d+)\)")

SCHEMA = """
CREATE TABLE city_article_graph (
    geoname_id INTEGER NOT NULL,
    seed_page_id INTEGER NOT NULL,
    article_page_id INTEGER NOT NULL,
    graph_distance INTEGER NOT NULL,
    PRIMARY KEY (geoname_id, article_page_id)
);
CREATE INDEX city_article_graph_article_idx
    ON city_article_graph(article_page_id);
CREATE INDEX city_article_graph_city_idx
    ON city_article_graph(geoname_id, graph_distance);
"""


def sql_title(value: bytes) -> str:
    return value.decode("utf-8", errors="replace").replace("''", "'")


def stream_inserts(path: Path, table: str):
    needle = f"INSERT INTO `{table}`".encode()
    with gzip.open(path, "rb") as source:
        for line in source:
            if line.startswith(needle):
                yield line


def build_lookup_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE page_titles(page_id INTEGER PRIMARY KEY, title TEXT NOT NULL);
        CREATE INDEX page_titles_title_idx ON page_titles(title);
        CREATE TABLE link_targets(lt_id INTEGER PRIMARY KEY, title TEXT NOT NULL);
        CREATE INDEX link_targets_title_idx ON link_targets(title);
        CREATE TABLE raw_edges(source_page_id INTEGER NOT NULL, target_lt_id INTEGER NOT NULL);
        """
    )
    rows = []
    for line in stream_inserts(PAGE_DUMP, "page"):
        for match in PAGE_RE.finditer(line):
            page_id, namespace, title, _is_redirect = match.groups()
            if int(namespace) == 0:
                rows.append((int(page_id), sql_title(title)))
        if len(rows) >= 50_000:
            connection.executemany("INSERT OR IGNORE INTO page_titles VALUES (?, ?)", rows)
            connection.commit()
            rows.clear()
    if rows:
        connection.executemany("INSERT OR IGNORE INTO page_titles VALUES (?, ?)", rows)
        rows.clear()
    connection.commit()

    rows = []
    for line in stream_inserts(LINKTARGET_DUMP, "linktarget"):
        for match in LINKTARGET_RE.finditer(line):
            target_id, namespace, title = match.groups()
            if int(namespace) == 0:
                rows.append((int(target_id), sql_title(title)))
        if len(rows) >= 50_000:
            connection.executemany("INSERT OR IGNORE INTO link_targets VALUES (?, ?)", rows)
            connection.commit()
            rows.clear()
    if rows:
        connection.executemany("INSERT OR IGNORE INTO link_targets VALUES (?, ?)", rows)
    connection.commit()


def build_graph(matches: Path, output: Path, city: str | None) -> None:
    frame = pd.read_parquet(matches)
    frame = frame[~frame["is_redirect"]].copy()
    if city:
        frame = frame[frame["canonical_query"].eq(city)]
    frame = frame.drop_duplicates(["geoname_id", "page_id"])
    if frame.empty:
        raise RuntimeError("No non-redirect city seeds matched the requested filter")

    seed_by_page: dict[int, list[int]] = {}
    for row in frame[["geoname_id", "page_id"]].itertuples(index=False):
        seed_by_page.setdefault(int(row.page_id), []).append(int(row.geoname_id))
    seed_ids = set(seed_by_page)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(SCHEMA)
    connection.execute("CREATE TABLE city_seeds(geoname_id INTEGER, page_id INTEGER)")
    connection.executemany(
        "INSERT INTO city_seeds VALUES (?, ?)",
        [(geoname_id, page_id) for page_id, ids in seed_by_page.items() for geoname_id in ids],
    )

    print("Scanning pagelinks for city seed pages")
    rows = []
    target_ids: set[int] = set()
    edge_count = 0
    for line in stream_inserts(PAGELINKS_DUMP, "pagelinks"):
        for match in PAGELINK_RE.finditer(line):
            source_page, source_namespace, target_id = map(int, match.groups())
            if source_namespace == 0 and source_page in seed_ids:
                rows.append((source_page, target_id))
                target_ids.add(target_id)
        if len(rows) >= 100_000:
            connection.executemany("INSERT INTO raw_edges VALUES (?, ?)", rows)
            edge_count += len(rows)
            connection.commit()
            rows.clear()
    if rows:
        connection.executemany("INSERT INTO raw_edges VALUES (?, ?)", rows)
        edge_count += len(rows)
    connection.commit()
    print(f"Seed link edges: {edge_count:,}")

    print(f"Resolving {len(target_ids):,} linked targets")
    target_titles: dict[int, str] = {}
    for line in stream_inserts(LINKTARGET_DUMP, "linktarget"):
        for match in LINKTARGET_RE.finditer(line):
            target_id, namespace, title = match.groups()
            target_id = int(target_id)
            if int(namespace) == 0 and target_id in target_ids:
                target_titles[target_id] = sql_title(title)

    target_title_set = set(target_titles.values())
    title_to_page: dict[str, int] = {}
    for line in stream_inserts(PAGE_DUMP, "page"):
        for match in PAGE_RE.finditer(line):
            page_id, namespace, title, _is_redirect = match.groups()
            if int(namespace) == 0:
                title_value = sql_title(title)
                if title_value in target_title_set:
                    page_id = int(page_id)
                    title_to_page[title_value] = min(
                        page_id,
                        title_to_page.get(title_value, page_id),
                    )

    connection.execute(
        """
        INSERT OR IGNORE INTO city_article_graph
            (geoname_id, seed_page_id, article_page_id, graph_distance)
        SELECT geoname_id, page_id, page_id, 0 FROM city_seeds
        """
    )
    graph_rows = []
    edge_cursor = connection.execute(
        "SELECT source_page_id, target_lt_id FROM raw_edges"
    )
    for source_page_id, target_id in edge_cursor:
        title = target_titles.get(int(target_id))
        article_page_id = title_to_page.get(title) if title else None
        if article_page_id is None or article_page_id == source_page_id:
            continue
        for geoname_id in seed_by_page.get(int(source_page_id), []):
            graph_rows.append((geoname_id, source_page_id, article_page_id, 1))
    connection.executemany(
        "INSERT OR IGNORE INTO city_article_graph VALUES (?, ?, ?, ?)",
        graph_rows,
    )
    connection.execute("DROP TABLE raw_edges")
    connection.execute("DROP TABLE city_seeds")
    connection.commit()
    count = connection.execute("SELECT count(*) FROM city_article_graph").fetchone()[0]
    city_count = connection.execute(
        "SELECT count(DISTINCT geoname_id) FROM city_article_graph"
    ).fetchone()[0]
    connection.close()
    temporary.replace(output)
    print(f"Saved {count:,} graph rows for {city_count:,} cities: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, default=MATCHES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--city", help="Canonical query to build, e.g. 'Paris, France'")
    args = parser.parse_args()
    for path in (args.matches, PAGE_DUMP, LINKTARGET_DUMP, PAGELINKS_DUMP):
        if not path.exists():
            raise FileNotFoundError(path)
    build_graph(args.matches, args.output, args.city)


if __name__ == "__main__":
    main()
