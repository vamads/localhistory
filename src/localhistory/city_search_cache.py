"""Read precomputed default search responses for recognized cities."""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache

from .search import DATA_DIR


CITY_SEARCH_CACHE_PATH = DATA_DIR / "city_search_cache.sqlite"
CITY_SEARCH_CACHE_VERSION = "1"
CITY_SEARCH_CACHE_RADIUS_KM = 50.0
CITY_SEARCH_CACHE_LIMIT = 150


@lru_cache(maxsize=1)
def _cache_available() -> bool:
    return CITY_SEARCH_CACHE_PATH.exists()


def load_cached_city_response(
    geoname_id: int,
    radius_km: float,
    limit: int,
    linked_only: bool,
) -> dict | None:
    """Return a cached response when the request matches the built snapshot."""
    if (
        linked_only
        or not _cache_available()
        or radius_km != CITY_SEARCH_CACHE_RADIUS_KM
        or limit != CITY_SEARCH_CACHE_LIMIT
    ):
        return None

    with sqlite3.connect(
        f"file:{CITY_SEARCH_CACHE_PATH}?mode=ro&immutable=1",
        uri=True,
    ) as connection:
        row = connection.execute(
            """
            SELECT response_json
            FROM city_search_cache
            WHERE geoname_id = ?
              AND cache_version = ?
              AND radius_km = ?
              AND result_limit = ?
            """,
            [
                int(geoname_id),
                CITY_SEARCH_CACHE_VERSION,
                CITY_SEARCH_CACHE_RADIUS_KM,
                CITY_SEARCH_CACHE_LIMIT,
            ],
        ).fetchone()
    return json.loads(row[0]) if row is not None else None
