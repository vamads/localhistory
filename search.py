"""
search.py

Hybrid local history search combining:
  1. Coordinate proximity   — articles within radius_km of user location
  2. BM25 retrieval          — fallback for articles without coordinates
  3. Scoring & ranking      — BM25, distance, entity class, hop, date

"""

import math
import os
import re
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
import torch

try:
    from .city_queries import precomputed_city_embedding, resolve_city_query
    from .citation_index import CITATION_HEURISTIC_VERSION
except ImportError:  # Support running search.py/profile_search.py as scripts.
    from city_queries import precomputed_city_embedding, resolve_city_query
    from citation_index import CITATION_HEURISTIC_VERSION

def resolve_data_dir() -> Path:
    """Return the data directory, allowing local or deployed configuration."""
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()
INDEX_PATH = DATA_DIR / "local_history_index.parquet"
SQLITE_SEARCH_PATH = DATA_DIR / "local_history_search.sqlite"
CITATION_INDEX_PATH = DATA_DIR / "local_history_citations.sqlite"
KALM_EMBEDDING_MATRIX_PATH = DATA_DIR / "kalm_embeddings.npy"
KALM_EMBEDDING_PAGE_IDS_PATH = DATA_DIR / "kalm_embedding_page_ids.npy"
KALM_MODEL_NAME = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
ARTICLE_COLUMNS = """
    articles.page_id,
    articles.title,
    articles.first_paragraph,
    articles.is_list_article,
    articles.lat,
    articles.lon,
    articles.country,
    articles.hop,
    articles.year,
    articles.entity_class
"""


@lru_cache(maxsize=16)
def load_candidate_index(
    location_name: str,
    user_lat: float,
    user_lon: float,
    radius_km: float,
) -> pd.DataFrame:
    """Load the union of geographically nearby and qualified text candidates."""
    require_current_index(
        SQLITE_SEARCH_PATH,
        INDEX_PATH,
        "python preprocessing/05_build_sqlite_search.py --overwrite",
    )
    text_candidates = load_sqlite_text_candidates(location_name)
    nearby_candidates = load_sqlite_coordinate_candidates(
        user_lat,
        user_lon,
        radius_km,
    )
    candidates = pd.concat([text_candidates, nearby_candidates]).drop_duplicates(
        subset="page_id",
        keep="first",
    )
    for column in (
        "is_list_article",
        "exact_title_match",
        "lead_exact_match",
        "exact_full_text_match",
    ):
        candidates[column] = candidates[column].astype(bool)
    return candidates.reset_index(drop=True)


def require_current_index(path: Path, source: Path, build_command: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing search index: {path}\nRun: {build_command}")
    if source.exists() and path.stat().st_mtime < source.stat().st_mtime:
        raise RuntimeError(f"Stale search index: {path}\nRun: {build_command}")


def quote_fts_phrase(value: str) -> str:
    return '"' + value.strip().lower().replace('"', '""') + '"'


def qualified_location_phrases(location_name: str) -> tuple[str, ...]:
    """Build exact FTS phrases for a city paired with state/country aliases."""
    parts = [part.strip() for part in location_name.split(",") if part.strip()]
    city = resolve_city_query(location_name)
    city_names = [parts[0]] if parts else []
    qualifiers = parts[1:]
    if city is not None:
        canonical_parts = [
            part.strip()
            for part in city.canonical_query.split(",")
            if part.strip()
        ]
        city_names.extend([city.name, canonical_parts[0]])
        qualifiers.extend(canonical_parts[1:])
        if city.admin1_name:
            qualifiers.extend([city.admin1_name, city.admin1_code])

    phrases = {
        quote_fts_phrase(f"{name} {qualifier}")
        for name in city_names
        for qualifier in qualifiers
        if name and qualifier
    }
    if not phrases:
        raise ValueError(f"A state or country is required to search for {location_name!r}")
    return tuple(sorted(phrases))


@lru_cache(maxsize=32)
def load_citation_counts(location_name: str) -> dict[int, int]:
    """Return citation-sentence matches from the required FTS5 index."""
    require_current_index(
        CITATION_INDEX_PATH,
        SQLITE_SEARCH_PATH,
        "python preprocessing/08_build_citation_index.py --overwrite",
    )

    database_uri = f"file:{CITATION_INDEX_PATH}?mode=ro&immutable=1"
    query = """
        SELECT
            CAST(page_id AS INTEGER) AS page_id,
            MIN(2, count(*)) AS context_count
        FROM citation_fts
        WHERE citation_fts MATCH ?
        GROUP BY page_id
    """
    with sqlite3.connect(database_uri, uri=True) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        source_stat = SQLITE_SEARCH_PATH.stat()
        if (
            metadata.get("citation_heuristic_version")
            != CITATION_HEURISTIC_VERSION
            or metadata.get("source_size") != str(source_stat.st_size)
            or metadata.get("source_mtime_ns") != str(source_stat.st_mtime_ns)
        ):
            raise RuntimeError(
                f"Stale citation index: {CITATION_INDEX_PATH}\n"
                "Run: python preprocessing/08_build_citation_index.py --overwrite"
            )
        rows = connection.execute(
            query,
            [quote_fts_phrase(location_name.split(",", 1)[0])],
        ).fetchall()
    return {int(page_id): int(count) for page_id, count in rows}


def load_sqlite_text_candidates(location_name: str) -> pd.DataFrame:
    """Retrieve articles containing a qualified place phrase."""
    database_uri = f"file:{SQLITE_SEARCH_PATH}?mode=ro&immutable=1"
    phrases = qualified_location_phrases(location_name)
    match_query = " OR ".join(phrases)
    query = f"""
        SELECT
            {ARTICLE_COLUMNS},
            -bm25(article_fts, 10.0, 3.0, 1.0) AS bm25_score
        FROM article_fts
        JOIN articles ON articles.page_id = article_fts.rowid
        WHERE article_fts MATCH ?
          AND NOT articles.is_redirect
        ORDER BY articles.page_id
    """
    with sqlite3.connect(database_uri, uri=True) as connection:
        candidates = pd.read_sql_query(
            query,
            connection,
            params=[match_query],
        )
        for fts_column, result_column in (
            ("title", "exact_title_match"),
            ("first_paragraph", "lead_exact_match"),
            ("full_text", "exact_full_text_match"),
        ):
            matches = connection.execute(
                "SELECT rowid FROM article_fts WHERE article_fts MATCH ?",
                [" OR ".join(f"{fts_column} : {phrase}" for phrase in phrases)],
            )
            page_ids = {page_id for (page_id,) in matches}
            candidates[result_column] = candidates["page_id"].isin(page_ids)

    return candidates


def load_sqlite_coordinate_candidates(
    user_lat: float,
    user_lon: float,
    radius_km: float,
) -> pd.DataFrame:
    """Retrieve a bounding box around the search point without a text filter."""
    latitude_delta = radius_km / 111.0
    longitude_delta = radius_km / max(
        1.0,
        111.0 * abs(math.cos(math.radians(user_lat))),
    )
    longitude_clause = "articles.lon BETWEEN ? AND ?"
    parameters = [
        user_lat - latitude_delta,
        user_lat + latitude_delta,
        user_lon - longitude_delta,
        user_lon + longitude_delta,
    ]
    if parameters[2] < -180 or parameters[3] > 180:
        longitude_clause = "1 = 1"
        parameters = parameters[:2]

    query = f"""
        SELECT
            {ARTICLE_COLUMNS},
            0.0 AS bm25_score,
            0 AS exact_title_match,
            0 AS lead_exact_match,
            0 AS exact_full_text_match
        FROM articles
        WHERE NOT articles.is_redirect
          AND articles.lat BETWEEN ? AND ?
          AND {longitude_clause}
    """
    database_uri = f"file:{SQLITE_SEARCH_PATH}?mode=ro&immutable=1"
    with sqlite3.connect(database_uri, uri=True) as connection:
        return pd.read_sql_query(query, connection, params=parameters)


def kalm_device() -> str:
    """Select the fastest available PyTorch device for embedding searches."""
    configured = os.getenv("LOCAL_HISTORY_EMBEDDING_DEVICE")
    if configured:
        return configured
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@lru_cache(maxsize=1)
def load_kalm_model():
    """Load the same KaLM model used to create the stored embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        KALM_MODEL_NAME,
        trust_remote_code=True,
        device=kalm_device(),
    )
    model.max_seq_length = 4096
    return model


@lru_cache(maxsize=1)
def load_kalm_embeddings():
    """Memory-map embeddings and their sorted page IDs without loading vectors."""
    if not (
        KALM_EMBEDDING_MATRIX_PATH.exists()
        and KALM_EMBEDDING_PAGE_IDS_PATH.exists()
    ):
        raise FileNotFoundError(
            "Missing memory-mapped embeddings. Run: "
            "python preprocessing/06_build_embedding_memmap.py --overwrite"
        )

    matrix = np.load(KALM_EMBEDDING_MATRIX_PATH, mmap_mode="r")
    page_ids = np.load(KALM_EMBEDDING_PAGE_IDS_PATH, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, found {matrix.shape}")
    if page_ids.ndim != 1 or len(page_ids) != len(matrix):
        raise ValueError(
            f"Embedding shapes do not match: {page_ids.shape} and {matrix.shape}"
        )
    if len(page_ids) > 1 and np.any(page_ids[1:] <= page_ids[:-1]):
        raise ValueError("Embedding page IDs must be unique and sorted")
    return matrix, page_ids


@lru_cache(maxsize=16)
def kalm_scores(
    location_name: str,
    target_page_ids: tuple[int, ...],
) -> pd.Series:
    """Return cosine similarity to each stored first-paragraph embedding."""
    embedding_matrix, embedding_page_ids = load_kalm_embeddings()
    if len(embedding_page_ids) == 0:
        return pd.Series(dtype="float32")

    query_embedding = precomputed_city_embedding(location_name)
    if query_embedding is None:
        model = load_kalm_model()
        query_embedding = model.encode(
            [location_name],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
    if query_embedding.shape != (embedding_matrix.shape[1],):
        raise ValueError(
            "Query and article embedding dimensions do not match: "
            f"{query_embedding.shape} and {embedding_matrix.shape}"
        )
    device = kalm_device()
    query = torch.from_numpy(query_embedding).to(device)

    requested_page_ids = np.fromiter(target_page_ids, dtype=np.int64)
    rows = np.searchsorted(embedding_page_ids, requested_page_ids)
    valid = rows < len(embedding_page_ids)
    valid[valid] &= embedding_page_ids[rows[valid]] == requested_page_ids[valid]
    rows = rows[valid]
    selected_page_ids = requested_page_ids[valid]
    if len(rows) == 0:
        return pd.Series(dtype="float32")

    # Fancy indexing materializes only selected rows from the memory map. The
    # complete 1.5 GiB matrix remains on disk instead of being uploaded to MPS.
    candidate_matrix = np.array(
        embedding_matrix[rows],
        dtype=np.float32,
        order="C",
        copy=True,
    )
    candidate_tensor = torch.from_numpy(candidate_matrix).to(device)
    with torch.inference_mode():
        selected_scores = (candidate_tensor @ query).cpu().numpy()
    return pd.Series(
        dict(zip(selected_page_ids.tolist(), selected_scores.astype(float))),
        dtype="float32",
    )


def normalize_scores(values: pd.Series) -> pd.Series:
    """Min-max normalize one query's scores to [0, 1]."""
    minimum = values.min()
    maximum = values.max()
    if pd.isna(minimum) or maximum <= minimum:
        return pd.Series(0.0, index=values.index)
    return (values - minimum) / (maximum - minimum)


def place_occurrence_features(
    frame: pd.DataFrame,
    location_name: str,
) -> pd.DataFrame:
    """Score where and how often the city appears in title and lead text."""
    city = location_name.split(",", 1)[0].strip().casefold()
    if not city:
        frame["place_title_mentions"] = 0
        frame["place_lead_mentions"] = 0
        frame["place_position_score"] = 0.0
        return frame

    pattern = re.compile(rf"(?<!\w){re.escape(city)}(?!\w)")
    titles = frame["title"].fillna("").astype(str).str.casefold()
    leads = frame["first_paragraph"].fillna("").astype(str).str.casefold()
    frame["place_title_mentions"] = titles.str.count(pattern)
    frame["place_lead_mentions"] = leads.str.count(pattern)

    def first_position(value: str) -> float:
        match = pattern.search(value)
        if match is None:
            return 0.0
        return 1.0 - (match.start() / max(len(value), 1))

    frame["place_position_score"] = leads.map(first_position)
    return frame


# ── Main search ───────────────────────────────────────────────────────────────


def search_local_history(
    location_name: str,
    user_lat: float,
    user_lon: float,
    radius_km: float = 50.0,
    top_n: int = 30,
    bm25_min_score: float = 1.0,
    kalm_top_k: int = 200,
    index: pd.DataFrame = None,
) -> dict:
    if index is None:
        index = load_candidate_index(
            location_name,
            user_lat,
            user_lon,
            radius_km,
        )

    df = index.copy()

    df = place_occurrence_features(df, location_name)

    df["bm25_score"] = pd.to_numeric(
        df["bm25_score"], errors="coerce"
    ).fillna(0.0)
    df["exact_score"] = (
        10 * df["exact_title_match"].astype(float)
        + 3 * df["exact_full_text_match"].astype(float)
    )
    text_match = (
        df["exact_title_match"]
        | df["lead_exact_match"]
        | df["exact_full_text_match"]
    )
    citation_counts = load_citation_counts(location_name)
    df["citation_context_count"] = (
        df["page_id"].map(citation_counts).fillna(0).astype("int8")
    )
    # Citation context is a weak negative signal, not a per-reference
    # subtraction. Strong title/lead evidence means the article is topical,
    # even if it contains many ordinary bibliographic references.
    df["citation_penalty"] = np.where(
        df["exact_title_match"] | df["lead_exact_match"],
        0.0,
        np.minimum(8.0, 4.0 * df["citation_context_count"]),
    )

    # KaLM embeddings represent first paragraphs (see kalm_embeddings.py).
    kalm_by_page_id = kalm_scores(
        location_name,
        tuple(df.loc[text_match, "page_id"].tolist()),
    )
    df["kalm_score"] = df["page_id"].map(kalm_by_page_id).fillna(0.0)
    df["bm25_score_normalized"] = 0.0
    df["kalm_score_normalized"] = 0.0
    df.loc[text_match, "bm25_score_normalized"] = normalize_scores(
        df.loc[text_match, "bm25_score"]
    )
    df.loc[text_match, "kalm_score_normalized"] = normalize_scores(
        df.loc[text_match, "kalm_score"]
    )

    # Layer 1: coordinate proximity. The candidate index already contains
    # coordinates, so do not reload the full coordinate corpus.
    coords = df[["page_id", "lat", "lon"]].dropna(subset=["lat", "lon"])
    lat1 = math.radians(user_lat)
    lat2 = np.radians(coords["lat"].to_numpy(dtype=np.float64))
    dlat = lat2 - lat1
    dlambda = np.radians(coords["lon"].to_numpy(dtype=np.float64) - user_lon)
    a = (
        np.sin(dlat / 2) ** 2
        + math.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2) ** 2
    )
    distances = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_by_page_id = pd.Series(
        distances,
        index=coords["page_id"].to_numpy(),
        dtype="float64",
    )
    df["distance_km"] = df["page_id"].map(distance_by_page_id)
    nearby = df[df["distance_km"].notna() & (df["distance_km"] <= radius_km)].copy()
    nearby["source"] = "coordinates"

    # Layer 2: BM25 fallback for articles without coordinate candidates.
    # BM25 scores individual terms, so require the complete city phrase here.
    nearby_ids = set(nearby["page_id"])
    bm25_matches = df[
        (df["bm25_score"] >= bm25_min_score)
        & text_match
        & ~df["page_id"].isin(nearby_ids)
        & df["lat"].isna()
    ].copy()
    bm25_matches["distance_km"] = np.nan
    bm25_matches["source"] = "bm25"

    # Add exact-title and semantic candidates that BM25 may miss.
    ranked_kalm = df[text_match].nlargest(kalm_top_k, "kalm_score")
    extra_matches = df[
        (
            df["exact_title_match"]
            | df["page_id"].isin(ranked_kalm["page_id"])
        )
        & ~df["page_id"].isin(nearby_ids)
        & df["lat"].isna()
    ].copy()
    extra_matches["distance_km"] = np.nan
    extra_matches["source"] = "exact_or_kalm"

    # Coordinate candidates take precedence; BM25 supplies the missing ones.
    combined = pd.concat([nearby, bm25_matches, extra_matches]).drop_duplicates(
        subset="page_id", keep="first"
    )

    score = combined["exact_score"].astype(float)
    score -= combined["citation_penalty"]
    score += 5 * combined["bm25_score_normalized"]
    score += 2 * combined["kalm_score_normalized"]
    score += (
        8 * combined["place_title_mentions"]
        + 5 * combined["place_lead_mentions"]
        + 2 * np.log1p(
            combined["place_title_mentions"]
            + combined["place_lead_mentions"]
        )
        + 4 * combined["place_position_score"]
    )
    # Geography is context, not relevance. Keep it as a small tie-breaker so
    # a strong text match can outrank a weak coordinate-only match.
    score += (
        0.5 * (1 - combined["distance_km"] / max(radius_km, 1.0))
    ).clip(lower=0).fillna(0)
    score += ((5 - combined["hop"]) * 2).fillna(0)
    score += combined["entity_class"].map(
        {"event": 4, "place": 3, "person": 2, "work": 1, "organization": 2}
    ).fillna(0)
    score += 2 * combined["year"].notna()
    score -= 5 * combined["is_list_article"].astype(float)
    combined["score"] = score

    # Coordinates are a hard inclusion rule, not a ranking rule. Keep every
    # coordinate-backed result, add the best text-only results, and then rank
    # the combined set by the same text-led score.
    coordinate_results = combined[combined["distance_km"].notna()]
    text_results = combined[combined["distance_km"].isna()].nlargest(
        top_n,
        "score",
    )
    results = pd.concat(
        [coordinate_results, text_results.head(top_n)],
        ignore_index=True,
    ).sort_values("score", ascending=False)

    return {
        "results": results,
        "stats": {
            "total_found": len(combined),
            "from_coords": (combined["source"] == "coordinates").sum(),
        },
    }
