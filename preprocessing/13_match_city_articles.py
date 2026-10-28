"""Match GeoNames cities to Wikipedia article pages in the page dump.

This is intentionally a cheap first pass. It scans ``page.sql.gz`` rather than
the much larger article XML dump and writes page IDs that can later be used to
extract article text and link neighborhoods.

Usage::

    uv run --extra preprocessing python preprocessing/13_match_city_articles.py

The output is ``city_article_matches.parquet`` in ``LOCAL_HISTORY_DATA_DIR``.
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
CITY_DATABASE = DATA_DIR / "city_queries.sqlite"
PAGE_DUMP = DATA_DIR / "enwiki-latest-page.sql.gz"
OUTPUT = DATA_DIR / "city_article_matches.parquet"

# page.sql format begins with:
#   (page_id,page_namespace,'page_title',page_is_redirect,...)
PAGE_ROW = re.compile(
    rb"\((\d+),(\d+),'((?:''|[^'])*)',(\d+),"
)


def sql_title(value: bytes) -> str:
    return value.decode("utf-8", errors="replace").replace("''", "'")


def title_key(value: str) -> str:
    return " ".join(value.replace("_", " ").casefold().split())


def city_title_candidates(row: sqlite3.Row) -> dict[str, str]:
    """Return normalized Wikipedia title candidates and their match type."""
    name = str(row["name"]).strip()
    country = str(row["country_name"]).strip()
    country_code = str(row["country_code"]).strip()
    state = str(row["admin1_name"]).strip()
    state_code = str(row["admin1_code"]).strip()
    candidates = {title_key(name): "exact_name"}

    # Wikipedia commonly disambiguates cities with a state, province, or
    # country in parentheses. Keep these candidates bounded and deterministic.
    qualifiers = [
        (state, "state_disambiguation"),
        (state_code, "state_code_disambiguation"),
    ]
    # ``City, United States`` is too generic to identify a particular US
    # place. Country-qualified titles remain useful for non-US cities.
    if country_code != "US":
        qualifiers.extend(
            [
                (country, "country_disambiguation"),
                (country_code, "country_code_disambiguation"),
            ]
        )
    for qualifier, match_type in qualifiers:
        if qualifier:
            candidates.setdefault(
                title_key(f"{name} ({qualifier})"),
                match_type,
            )
    comma_qualifiers = [
        (state, "comma_state"),
        (state_code, "comma_state_code"),
    ]
    if country_code != "US":
        comma_qualifiers.extend(
            [
                (country, "comma_country"),
                (country_code, "comma_country_code"),
            ]
        )
    for qualifier, match_type in comma_qualifiers:
        if qualifier:
            candidates.setdefault(
                title_key(f"{name}, {qualifier}"),
                match_type,
            )
    return candidates


def load_cities(path: Path) -> dict[str, list[dict]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT geoname_id, name, canonical_query, country_code,
                   country_name, admin1_code, admin1_name, population
            FROM cities
            ORDER BY geoname_id
            """
        ).fetchall()
    finally:
        connection.close()

    by_title: dict[str, list[dict]] = {}
    for row in rows:
        for candidate, match_type in city_title_candidates(row).items():
            by_title.setdefault(candidate, []).append(
                {
                    "geoname_id": int(row["geoname_id"]),
                    "city_name": str(row["name"]),
                    "canonical_query": str(row["canonical_query"]),
                    "country_code": str(row["country_code"]),
                    "country_name": str(row["country_name"]),
                    "admin1_code": str(row["admin1_code"]),
                    "admin1_name": str(row["admin1_name"]),
                    "population": int(row["population"]),
                    "match_type": match_type,
                }
            )
    return by_title


def match_pages(page_dump: Path, cities: dict[str, list[dict]]) -> pd.DataFrame:
    matches: list[dict] = []
    seen: set[tuple[int, int]] = set()
    with gzip.open(page_dump, "rb") as source:
        for line_number, line in enumerate(source, start=1):
            for match in PAGE_ROW.finditer(line):
                page_id = int(match.group(1))
                namespace = int(match.group(2))
                if namespace != 0:
                    continue
                title = sql_title(match.group(3))
                candidates = cities.get(title_key(title))
                if not candidates:
                    continue
                # A bare title such as ``Paris`` or ``Detroit`` can refer to
                # many GeoNames places. Use it for the largest matching city;
                # smaller same-name cities must match a disambiguated title.
                if all(city["match_type"] == "exact_name" for city in candidates):
                    largest_population = max(
                        city["population"] for city in candidates
                    )
                    candidates = [
                        city
                        for city in candidates
                        if city["population"] == largest_population
                    ]
                is_redirect = bool(int(match.group(4)))
                for city in candidates:
                    key = (city["geoname_id"], page_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append(
                        {
                            **city,
                            "page_id": page_id,
                            "wikipedia_title": title,
                            "is_redirect": is_redirect,
                        }
                    )
            if line_number % 1_000_000 == 0:
                print(
                    f"Scanned {line_number:,} SQL lines; "
                    f"matched {len(matches):,} city-page rows",
                    flush=True,
                )
    return pd.DataFrame(matches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-database", type=Path, default=CITY_DATABASE)
    parser.add_argument("--page-dump", type=Path, default=PAGE_DUMP)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not args.city_database.exists():
        raise FileNotFoundError(args.city_database)
    if not args.page_dump.exists():
        raise FileNotFoundError(args.page_dump)

    cities = load_cities(args.city_database)
    print(f"Loaded {sum(len(rows) for rows in cities.values()):,} title variants")
    frame = match_pages(args.page_dump, cities)
    if frame.empty:
        raise RuntimeError("No city article matches found")
    frame = frame.sort_values(
        ["geoname_id", "match_type", "is_redirect", "page_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    matched_cities = frame["geoname_id"].nunique()
    print(f"Matched {matched_cities:,} cities")
    print(f"Matched {len(frame):,} city-page rows")
    print(f"Output: {args.output}")
    print("Examples:")
    print(
        frame[
            frame["city_name"].str.casefold().isin(
                {"paris", "detroit", "ann arbor", "south lyon", "new york city"}
            )
        ][
            ["city_name", "canonical_query", "wikipedia_title", "page_id", "match_type"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
