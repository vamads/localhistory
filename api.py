"""HTTP API for the Local History web client."""

from functools import lru_cache
import math
import os
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .search import INDEX_PATH, search_local_history


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


app = FastAPI(title="Local History API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("LOCAL_HISTORY_WEB_ORIGIN", "http://localhost:5173")],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def article_index() -> pd.DataFrame:
    return pd.read_parquet(INDEX_PATH)


@lru_cache(maxsize=128)
def geocode(place: str) -> Location:
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
    if row.get("source") == "coordinates":
        return "Mapped nearby"
    if row.get("exact_title_match", False):
        return "City in title"
    if row.get("lead_exact_match", False):
        return "City in introduction"
    if row.get("citation_penalty", 0) > 0:
        return "Article mention; citation context detected"
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
    q: str = Query(min_length=2),
    radius_km: float = Query(default=50, gt=0, le=500),
    limit: int = Query(default=150, ge=10, le=500),
) -> SearchResponse:
    try:
        location = await run_in_threadpool(geocode, q)
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    output = await run_in_threadpool(
        search_local_history,
        location_name=q,
        user_lat=location.latitude,
        user_lon=location.longitude,
        radius_km=radius_km,
        top_n=limit,
        bm25_min_score=1.0,
        kalm_top_k=200,
        worst_n=0,
        fallback_top_n=0,
        index=article_index(),
    )
    rows = output["results"]
    mapped = [
        card_from_row(row)
        for _, row in rows[rows["distance_km"].notna()].iterrows()
    ]
    connected = [
        card_from_row(row)
        for _, row in rows[rows["distance_km"].isna()].iterrows()
    ]
    stats = output["stats"]
    return SearchResponse(
        query=location,
        mapped=mapped,
        connected=connected,
        counts={
            "mapped": int(stats["from_coords"]),
            "connected": int(stats["total_found"] - stats["from_coords"]),
        },
    )


@app.get("/api/articles/{page_id}", response_model=ArticleDetail)
def article(page_id: int) -> ArticleDetail:
    rows = article_index().loc[lambda frame: frame["page_id"] == page_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Article not found")
    row = rows.iloc[0]
    country = clean_text(row.get("country")) or None
    return ArticleDetail(
        page_id=page_id,
        title=clean_text(row["title"]),
        first_paragraph=clean_text(row.get("first_paragraph")),
        full_text=clean_text(row.get("full_text")),
        entity_class=clean_text(row.get("entity_class"), "other"),
        year=finite_int(row.get("year")),
        country=country,
    )
