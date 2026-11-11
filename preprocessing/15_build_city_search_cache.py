"""Precompute default search responses for a population-filtered city set.

The job is resumable: existing city rows are skipped unless ``--overwrite`` is
passed. It stores the serialized API response, so the runtime does not need to
load the retrieval models for cached city searches.

Example::

    python preprocessing/15_build_city_search_cache.py --min-population 100000
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from localhistory.api import Location, SearchResponse, card_from_row  # noqa: E402
from localhistory.city_search_cache import (  # noqa: E402
    CITY_SEARCH_CACHE_LIMIT,
    CITY_SEARCH_CACHE_PATH,
    CITY_SEARCH_CACHE_RADIUS_KM,
    CITY_SEARCH_CACHE_VERSION,
)
from localhistory.city_queries import CITY_QUERY_DATABASE_PATH  # noqa: E402
from localhistory.search import (  # noqa: E402
    load_cross_encoder,
    load_faiss_index,
    search_local_history,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS city_search_cache (
    geoname_id INTEGER PRIMARY KEY,
    canonical_query TEXT NOT NULL,
    population INTEGER NOT NULL,
    country_name TEXT NOT NULL,
    radius_km REAL NOT NULL,
    result_limit INTEGER NOT NULL,
    cache_version TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    response_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS city_search_cache_version_idx
    ON city_search_cache(cache_version, radius_km, result_limit);
"""


def build_response(city, output) -> dict:
    rows = output["results"]
    mapped = [
        card_from_row(row).model_dump(mode="json")
        for _, row in rows[rows["distance_km"].notna()].iterrows()
    ]
    connected = [card_from_row(row).model_dump(mode="json") for _, row in rows.iterrows()]
    stats = output["stats"]
    return SearchResponse(
        query=Location(
            label=city.canonical_query,
            latitude=city.latitude,
            longitude=city.longitude,
        ),
        mapped=mapped,
        connected=connected,
        counts={
            "mapped": int(stats["from_coords"]),
            "connected": int(stats["total_found"]),
        },
    ).model_dump(mode="json")


def load_cities(min_population: int) -> list:
    query = """
        SELECT geoname_id, name, canonical_query, latitude, longitude,
               country_code, country_name, admin1_code, admin1_name,
               population, embedding_row
        FROM cities
        WHERE population >= ?
        ORDER BY population DESC, geoname_id
    """
    from localhistory.city_queries import CityQuery

    with sqlite3.connect(CITY_QUERY_DATABASE_PATH) as connection:
        rows = connection.execute(query, [min_population]).fetchall()
    return [CityQuery(*row) for row in rows]


def build_cache(
    min_population: int,
    batch_size: int,
    overwrite: bool,
    max_cities: int | None = None,
) -> None:
    cities = load_cities(min_population)
    if max_cities is not None:
        cities = cities[:max_cities]
    if not cities:
        raise RuntimeError(f"No cities found with population >= {min_population:,}")

    CITY_SEARCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CITY_SEARCH_CACHE_PATH)
    connection.executescript(SCHEMA)
    if overwrite:
        connection.execute(
            "DELETE FROM city_search_cache WHERE cache_version = ?",
            [CITY_SEARCH_CACHE_VERSION],
        )
        connection.commit()

    cached_ids = {
        row[0]
        for row in connection.execute(
            """
            SELECT geoname_id
            FROM city_search_cache
            WHERE cache_version = ?
              AND radius_km = ?
              AND result_limit = ?
            """,
            [
                CITY_SEARCH_CACHE_VERSION,
                CITY_SEARCH_CACHE_RADIUS_KM,
                CITY_SEARCH_CACHE_LIMIT,
            ],
        )
    }
    pending = [city for city in cities if city.geoname_id not in cached_ids]
    print(
        f"Cities selected: {len(cities):,}; cached: {len(cached_ids):,}; "
        f"remaining: {len(pending):,}"
    )
    print("Loading FAISS and cross-encoder...")
    load_faiss_index()
    encoder = load_cross_encoder()
    encoder.predict(
        [("warmup query", "warmup article")],
        batch_size=1,
        show_progress_bar=False,
    )

    started = time.perf_counter()
    try:
        for index, city in enumerate(pending, start=1):
            city_started = time.perf_counter()
            output = search_local_history(
                location_name=city.canonical_query,
                user_lat=city.latitude,
                user_lon=city.longitude,
                radius_km=CITY_SEARCH_CACHE_RADIUS_KM,
                top_n=CITY_SEARCH_CACHE_LIMIT,
                bm25_min_score=1.0,
                dense_top_k=200,
                population=city.population,
                country_name=city.country_name,
                geoname_id=city.geoname_id,
                linked_only=False,
                rerank=True,
            )
            response_json = json.dumps(
                build_response(city, output),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO city_search_cache(
                    geoname_id, canonical_query, population, country_name,
                    radius_km, result_limit, cache_version, generated_at_utc,
                    response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    city.geoname_id,
                    city.canonical_query,
                    city.population,
                    city.country_name,
                    CITY_SEARCH_CACHE_RADIUS_KM,
                    CITY_SEARCH_CACHE_LIMIT,
                    CITY_SEARCH_CACHE_VERSION,
                    datetime.now(timezone.utc).isoformat(),
                    response_json,
                ],
            )
            if index % batch_size == 0:
                connection.commit()
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            print(
                f"[{index:,}/{len(pending):,}] {city.canonical_query}: "
                f"{time.perf_counter() - city_started:.2f}s; "
                f"{rate:.2f} cities/s",
                flush=True,
            )
        connection.commit()
    finally:
        connection.close()
    print(f"Saved cache: {CITY_SEARCH_CACHE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-population", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-cities", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_cache(
        args.min_population,
        max(1, args.batch_size),
        args.overwrite,
        args.max_cities,
    )


if __name__ == "__main__":
    main()
