"""Build a bounded Wikipedia link graph for matched city articles.

This is intentionally not a complete Wikipedia graph. It follows links from
the matched city article for a bounded number of hops and stores one row per
city/article pair at the shortest distance. The parent page is retained so a
notebook can explain paths such as Paris -> French Revolution -> Robespierre.

Examples::

    uv run --extra preprocessing python preprocessing/14_build_city_link_graph.py \
        --city "Paris, France" --max-hops 2

    uv run --extra preprocessing python preprocessing/14_build_city_link_graph.py \
        --max-hops 1

The second command is much cheaper for a broad all-city build. For exploration
of a few large cities, two or three hops is generally a better tradeoff.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sqlite3
import subprocess

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
PAGE_DUMP = DATA_DIR / "enwiki-latest-page.sql.gz"
LINKTARGET_DUMP = DATA_DIR / "enwiki-latest-linktarget.sql.gz"
PAGELINKS_DUMP = DATA_DIR / "enwiki-latest-pagelinks.sql.gz"
OUTPUT = DATA_DIR / "city_link_graph.sqlite"
MATCHES = DATA_DIR / "city_article_matches.parquet"

PAGE_RE = re.compile(rb"\((\d+),(\d+),'((?:''|[^'])*)',(\d+),")
LINKTARGET_RE = re.compile(rb"\((\d+),(\-?\d+),'((?:''|[^'])*)'")
PAGELINK_RE = re.compile(rb"\((\d+),(\-?\d+),(\d+)\)")

SCHEMA = """
CREATE TABLE city_article_graph (
    geoname_id INTEGER NOT NULL,
    seed_page_id INTEGER NOT NULL,
    article_page_id INTEGER NOT NULL,
    graph_distance INTEGER NOT NULL,
    parent_page_id INTEGER,
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
    """Yield lines belonging to INSERT statements in the MariaDB dump.

    Wikimedia's dump format places ``INSERT INTO ... VALUES`` on one line and
    the value tuples on one or more following lines, so checking only the
    header line silently produces an empty graph.
    """
    needle = f"INSERT INTO `{table}`".encode()
    inside_insert = False
    # Some locally downloaded Wikimedia dumps contain valid gzip data followed
    # by harmless trailing bytes. The system gzip decoder can recover the valid
    # stream; Python's gzip module raises BadGzipFile at the trailing bytes.
    process = subprocess.Popen(
        ["gzip", "-cd", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if line.startswith(needle):
                inside_insert = True
            if inside_insert:
                yield line
                if line.rstrip().endswith(b";"):
                    inside_insert = False
    finally:
        process.stdout.close()
        process.wait()


def resolve_targets(target_ids: set[int]) -> dict[int, int]:
    """Resolve a small set of pagelink target IDs to article page IDs."""
    if not target_ids:
        return {}

    target_titles: dict[int, str] = {}
    for line in stream_inserts(LINKTARGET_DUMP, "linktarget"):
        for match in LINKTARGET_RE.finditer(line):
            target_id, namespace, title = match.groups()
            target_id = int(target_id)
            if int(namespace) == 0 and target_id in target_ids:
                target_titles[target_id] = sql_title(title)

    wanted_titles = set(target_titles.values())
    title_to_page: dict[str, int] = {}
    for line in stream_inserts(PAGE_DUMP, "page"):
        for match in PAGE_RE.finditer(line):
            page_id, namespace, title, _is_redirect = match.groups()
            if int(namespace) != 0:
                continue
            title_value = sql_title(title)
            if title_value in wanted_titles:
                page_id = int(page_id)
                title_to_page[title_value] = min(
                    page_id, title_to_page.get(title_value, page_id)
                )

    return {
        target_id: title_to_page[title]
        for target_id, title in target_titles.items()
        if title in title_to_page
    }


def build_graph(
    matches: Path,
    output: Path,
    city: str | None,
    max_hops: int,
) -> None:
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")

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

    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(SCHEMA)

    seed_rows = [
        (geoname_id, page_id, page_id, 0, None)
        for page_id, geoname_ids in seed_by_page.items()
        for geoname_id in geoname_ids
    ]
    connection.executemany(
        "INSERT INTO city_article_graph VALUES (?, ?, ?, ?, ?)", seed_rows
    )

    # Compact breadth-first frontier: page_id -> (city, seed). We never
    # materialize every Wikipedia edge in a separate graph database.
    frontier: dict[int, set[tuple[int, int]]] = {
        page_id: {(geoname_id, page_id) for geoname_id in geoname_ids}
        for page_id, geoname_ids in seed_by_page.items()
    }
    visited: set[tuple[int, int, int]] = {
        (geoname_id, page_id, page_id)
        for page_id, geoname_ids in seed_by_page.items()
        for geoname_id in geoname_ids
    }

    for distance in range(1, max_hops + 1):
        frontier_pages = set(frontier)
        if not frontier_pages:
            break
        print(f"Scanning pagelinks for hop {distance} ({len(frontier_pages):,} pages)")

        edges: list[tuple[int, int]] = []
        target_ids: set[int] = set()
        for line in stream_inserts(PAGELINKS_DUMP, "pagelinks"):
            for match in PAGELINK_RE.finditer(line):
                source_page, source_namespace, target_id = map(int, match.groups())
                if source_namespace == 0 and source_page in frontier_pages:
                    edges.append((source_page, target_id))
                    target_ids.add(target_id)

        target_to_page = resolve_targets(target_ids)
        next_frontier: dict[int, set[tuple[int, int]]] = {}
        graph_rows = []
        for source_page, target_id in edges:
            article_page = target_to_page.get(target_id)
            if article_page is None or article_page == source_page:
                continue
            for geoname_id, seed_page in frontier[source_page]:
                visit = (geoname_id, seed_page, article_page)
                if visit in visited:
                    continue
                visited.add(visit)
                graph_rows.append(
                    (geoname_id, seed_page, article_page, distance, source_page)
                )
                next_frontier.setdefault(article_page, set()).add(
                    (geoname_id, seed_page)
                )

        connection.executemany(
            "INSERT OR IGNORE INTO city_article_graph VALUES (?, ?, ?, ?, ?)",
            graph_rows,
        )
        connection.commit()
        print(
            f"  Added {len(graph_rows):,} articles; "
            f"next frontier: {len(next_frontier):,} pages"
        )
        frontier = next_frontier

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
    parser.add_argument(
        "--max-hops", type=int, default=2,
        help="Maximum link distance from each city article (default: 2)",
    )
    args = parser.parse_args()
    for path in (args.matches, PAGE_DUMP, LINKTARGET_DUMP, PAGELINKS_DUMP):
        if not path.exists():
            raise FileNotFoundError(path)
    build_graph(args.matches, args.output, args.city, args.max_hops)


if __name__ == "__main__":
    main()
