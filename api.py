"""HTTP API for the Local History web client."""

from functools import lru_cache
import logging
import math
import os
import sqlite3
import time
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .city_queries import resolve_city_query
from .search import SQLITE_SEARCH_PATH, search_local_history


logger = logging.getLogger("localhistory.api")


class Location(BaseModel):
    label: str
    latitude: float
    longitude: float


class ArticleCard(BaseModel):
    page_id: int
    title: str
    first_paragraph: str
    entity_class: str
    year: int | None
    country: str | None
    latitude: float | None
    longitude: float | None
    distance_km: float | None
    score: float
    source: str
    match_reason: str


class SearchResponse(BaseModel):
    query: Location
    mapped: list[ArticleCard]
    connected: list[ArticleCard]
    counts: dict[str, int]


class ArticleDetail(BaseModel):
    page_id: int
    title: str
    first_paragraph: str
    full_text: str
    entity_class: str
    year: int | None
    country: str | None


class MapPoint(BaseModel):
    page_id: int
    latitude: float
    longitude: float


app = FastAPI(title="Local History API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("LOCAL_HISTORY_WEB_ORIGIN", "http://localhost:5173")],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def search_database() -> sqlite3.Connection:
    uri = f"file:{SQLITE_SEARCH_PATH}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


@lru_cache(maxsize=128)
def geocode(place: str) -> Location:
    local_city = resolve_city_query(place)
    if local_city is not None:
        return Location(
            label=local_city.canonical_query,
            latitude=local_city.latitude,
            longitude=local_city.longitude,
        )

    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "LocalHistoryExplorer/0.1"},
        timeout=10,
    )
    response.raise_for_status()
    matches = response.json()
    if not matches:
        raise ValueError(f"Could not find {place!r}")
    match = matches[0]
    return Location(
        label=match["display_name"],
        latitude=float(match["lat"]),
        longitude=float(match["lon"]),
    )


def finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def finite_int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def clean_text(value: Any, default: str = "") -> str:
    """Return display-safe text for nullable or collection-valued fields."""
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


def match_reason(row: pd.Series) -> str:
    if row.get("exact_title_match", False):
        return "City in title"
    if row.get("lead_exact_match", False):
        return "City in introduction"
    if row.get("citation_penalty", 0) > 0:
        return "Article mention; citation context detected"
    if row.get("source") == "coordinates":
        return "Mapped nearby"
    return "City mentioned in article"


def card_from_row(row: pd.Series) -> ArticleCard:
    country = clean_text(row.get("country")) or None
    return ArticleCard(
        page_id=int(row["page_id"]),
        title=clean_text(row["title"]),
        first_paragraph=clean_text(row.get("first_paragraph")),
        entity_class=clean_text(row.get("entity_class"), "other"),
        year=finite_int(row.get("year")),
        country=country,
        latitude=finite_float(row.get("lat")),
        longitude=finite_float(row.get("lon")),
        distance_km=finite_float(row.get("distance_km")),
        score=float(row.get("score", 0)),
        source=clean_text(row.get("source"), "unknown"),
        match_reason=match_reason(row),
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/search", response_model=SearchResponse)
async def search(
    response: Response,
    q: str = Query(min_length=2),
    radius_km: float = Query(default=50, gt=0, le=500),
    limit: int = Query(default=150, ge=10, le=500),
) -> SearchResponse:
    request_started = time.perf_counter()
    geocode_started = request_started
    local_city = resolve_city_query(q)
    try:
        location = await run_in_threadpool(geocode, q)
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    geocode_ms = (time.perf_counter() - geocode_started) * 1000

    search_started = time.perf_counter()
    output = await run_in_threadpool(
        search_local_history,
        location_name=(
            local_city.canonical_query if local_city is not None else location.label
        ),
        user_lat=location.latitude,
        user_lon=location.longitude,
        radius_km=radius_km,
        top_n=limit,
        bm25_min_score=1.0,
        kalm_top_k=200,
    )
    search_ms = (time.perf_counter() - search_started) * 1000

    serialization_started = time.perf_counter()
    rows = output["results"]
    mapped = [
        card_from_row(row)
        for _, row in rows[rows["distance_km"].notna()].iterrows()
    ]
    connected = [card_from_row(row) for _, row in rows.iterrows()]
    stats = output["stats"]
    result = SearchResponse(
        query=location,
        mapped=mapped,
        connected=connected,
        counts={
            "mapped": int(stats["from_coords"]),
            "connected": int(stats["total_found"]),
        },
    )
    serialization_ms = (time.perf_counter() - serialization_started) * 1000
    total_ms = (time.perf_counter() - request_started) * 1000
    response.headers["Server-Timing"] = (
        f"geocode;dur={geocode_ms:.1f}, "
        f"search;dur={search_ms:.1f}, "
        f"serialize;dur={serialization_ms:.1f}, "
        f"total;dur={total_ms:.1f}"
    )
    response.headers["X-Search-Time-Ms"] = f"{total_ms:.1f}"
    logger.info(
        "search q=%r geocode=%.1fms search=%.1fms serialize=%.1fms total=%.1fms",
        q,
        geocode_ms,
        search_ms,
        serialization_ms,
        total_ms,
    )
    return result


@app.get("/api/articles/{page_id}", response_model=ArticleDetail)
def article(page_id: int) -> ArticleDetail:
    with search_database() as connection:
        row = connection.execute(
            """
            SELECT title, first_paragraph, full_text, entity_class, year, country
            FROM articles
            WHERE page_id = ? AND NOT is_redirect
            """,
            [page_id],
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Article not found")
    title, first_paragraph, full_text, entity_class, year, country = row
    return ArticleDetail(
        page_id=page_id,
        title=clean_text(title),
        first_paragraph=clean_text(first_paragraph),
        full_text=clean_text(full_text),
        entity_class=clean_text(entity_class, "other"),
        year=finite_int(year),
        country=clean_text(country) or None,
    )


@app.get("/api/coordinates", response_model=list[MapPoint])
def coordinates() -> list[MapPoint]:
    """Return the coordinate layer once; the browser filters it by radius."""
    with search_database() as connection:
        rows = connection.execute(
            """
            SELECT page_id, lat, lon
            FROM articles
            WHERE NOT is_redirect AND lat IS NOT NULL AND lon IS NOT NULL
            """
        ).fetchall()
    return [
        MapPoint(page_id=int(page_id), latitude=float(lat), longitude=float(lon))
        for page_id, lat, lon in rows
    ]
