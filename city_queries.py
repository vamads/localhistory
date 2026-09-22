"""Resolve common city names to local coordinates and precomputed query vectors."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sqlite3
import unicodedata

import numpy as np


def resolve_data_dir() -> Path:
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()
CITY_QUERY_DATABASE_PATH = DATA_DIR / "city_queries.sqlite"
CITY_QUERY_EMBEDDINGS_PATH = DATA_DIR / "city_query_embeddings.npy"


@dataclass(frozen=True)
class CityQuery:
    geoname_id: int
    name: str
    canonical_query: str
    latitude: float
    longitude: float
    country_code: str
    country_name: str
    admin1_code: str
    admin1_name: str
    population: int
    embedding_row: int


def normalize_city_alias(value: str) -> str:
    """Normalize punctuation, case, spacing, and diacritics for alias lookup."""
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    characters = []
    for character in decomposed:
        if unicodedata.category(character) == "Mn":
            continue
        characters.append(character if character.isalnum() else " ")
    return " ".join("".join(characters).split())


def database_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro&immutable=1"


@lru_cache(maxsize=4_096)
def _resolve_normalized_city(normalized_alias: str) -> CityQuery | None:
    if not normalized_alias or not CITY_QUERY_DATABASE_PATH.exists():
        return None

    query = """
        SELECT
            cities.geoname_id,
            cities.name,
            cities.canonical_query,
            cities.latitude,
            cities.longitude,
            cities.country_code,
            cities.country_name,
            cities.admin1_code,
            cities.admin1_name,
            cities.population,
            cities.embedding_row
        FROM aliases
        JOIN cities USING (geoname_id)
        WHERE aliases.normalized_alias = ?
        ORDER BY cities.population DESC, cities.geoname_id
        LIMIT 1
    """
    with sqlite3.connect(
        database_uri(CITY_QUERY_DATABASE_PATH),
        uri=True,
    ) as connection:
        row = connection.execute(query, (normalized_alias,)).fetchone()
    return CityQuery(*row) if row is not None else None


def resolve_city_query(value: str) -> CityQuery | None:
    """Resolve an exact normalized city alias, preferring the largest match."""
    return _resolve_normalized_city(normalize_city_alias(value))


@lru_cache(maxsize=1)
def load_city_query_embeddings() -> np.ndarray | None:
    """Open precomputed city vectors as a read-only memory map."""
    if not CITY_QUERY_EMBEDDINGS_PATH.exists():
        return None
    embeddings = np.load(
        CITY_QUERY_EMBEDDINGS_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D city query embeddings, found shape {embeddings.shape}"
        )
    return embeddings


def precomputed_city_embedding(value: str) -> np.ndarray | None:
    """Return a copied city query vector, or None when no cache entry exists."""
    city = resolve_city_query(value)
    embeddings = load_city_query_embeddings()
    if city is None or embeddings is None:
        return None
    if not 0 <= city.embedding_row < len(embeddings):
        raise ValueError(
            f"Invalid embedding row {city.embedding_row} for {city.canonical_query}"
        )
    return np.array(embeddings[city.embedding_row], dtype=np.float32, copy=True)
