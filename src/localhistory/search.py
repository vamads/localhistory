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

try:
    from .city_queries import precomputed_city_embedding, resolve_city_query
    from .citation_index import CITATION_HEURISTIC_VERSION
except ImportError:  # Support running search.py/profile_search.py as scripts.
    from localhistory.city_queries import precomputed_city_embedding, resolve_city_query
    from localhistory.citation_index import CITATION_HEURISTIC_VERSION

def resolve_data_dir() -> Path:
    """Return the data directory, allowing local or deployed configuration."""
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parents[3] / "data",
        Path(__file__).resolve().parents[2] / "data",
        Path(__file__).resolve().parent / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()
INDEX_PATH = DATA_DIR / "local_history_index.parquet"
BASE_SEARCH_PATH = DATA_DIR / "local_history_search.sqlite"
RUNTIME_DATABASE_PATH = DATA_DIR / "local_history_runtime.sqlite"
SQLITE_SEARCH_PATH = (
    RUNTIME_DATABASE_PATH if RUNTIME_DATABASE_PATH.exists() else BASE_SEARCH_PATH
)
CITATION_INDEX_PATH = (
    SQLITE_SEARCH_PATH
    if SQLITE_SEARCH_PATH == RUNTIME_DATABASE_PATH
    else DATA_DIR / "local_history_citations.sqlite"
)
KALM_EMBEDDING_MATRIX_PATH = DATA_DIR / "kalm_embeddings.npy"
KALM_EMBEDDING_PAGE_IDS_PATH = DATA_DIR / "kalm_embedding_page_ids.npy"
LINK_COUNTS_PATH = DATA_DIR / "article_link_counts.parquet"
KALM_MODEL_NAME = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
ARTICLE_COLUMNS = """
    articles.page_id,
    articles.title,
    articles.first_paragraph,
    length(articles.first_paragraph) AS first_paragraph_length,
    length(articles.full_text) AS full_text_length,
    articles.is_list_article,
    articles.lat,
    articles.lon,
    articles.country,
    articles.wikidata_id,
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
    # The finalized runtime database is an immutable production snapshot. It
    # intentionally does not require the large Parquet source at query time.
    if path == RUNTIME_DATABASE_PATH:
        return
    if source.exists() and path.stat().st_mtime < source.stat().st_mtime:
        raise RuntimeError(f"Stale search index: {path}\nRun: {build_command}")


@lru_cache(maxsize=1)
def load_editorial_importance() -> pd.Series:
    """Load capped log incoming-link scores for optional secondary ranking.

    The link-count job scans all normal Wikipedia article links but stores
    only Local History targets. Missing link data is treated as a zero score
    so search remains usable while the one-time preprocessing job is running.
    """
    if RUNTIME_DATABASE_PATH.exists():
        database_uri = f"file:{RUNTIME_DATABASE_PATH}?mode=ro&immutable=1"
        with sqlite3.connect(database_uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT page_id, editorial_importance FROM article_importance"
            ).fetchall()
        if not rows:
            return pd.Series(dtype="float64", name="editorial_importance")
        return pd.Series(
            [float(score) for _, score in rows],
            index=[int(page_id) for page_id, _ in rows],
            name="editorial_importance",
        )

    if not LINK_COUNTS_PATH.exists():
        return pd.Series(dtype="float64", name="editorial_importance")

    counts = pd.read_parquet(
        LINK_COUNTS_PATH,
        columns=["page_id", "incoming_link_count"],
    )
    counts["page_id"] = pd.to_numeric(counts["page_id"], errors="coerce")
    counts["incoming_link_count"] = pd.to_numeric(
        counts["incoming_link_count"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    counts = counts.dropna(subset=["page_id"]).drop_duplicates("page_id")

    log_counts = np.log1p(counts["incoming_link_count"].to_numpy(dtype="float64"))
    cap = float(np.quantile(log_counts, 0.99)) if len(log_counts) else 0.0
    if cap <= 0.0:
        scores = np.zeros(len(counts), dtype="float64")
    else:
        scores = np.clip(log_counts / cap, 0.0, 1.0)

    return pd.Series(
        scores,
        index=counts["page_id"].astype("int64").to_numpy(),
        name="editorial_importance",
    )


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
        "python preprocessing/09_build_citation_index.py --overwrite",
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
        stale = metadata.get("citation_heuristic_version") != CITATION_HEURISTIC_VERSION
        if CITATION_INDEX_PATH != SQLITE_SEARCH_PATH:
            stale = stale or (
                metadata.get("source_size") != str(source_stat.st_size)
                or metadata.get("source_mtime_ns") != str(source_stat.st_mtime_ns)
            )
        if stale:
            raise RuntimeError(
                f"Stale citation index: {CITATION_INDEX_PATH}\n"
                "Run: python preprocessing/09_build_citation_index.py --overwrite"
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


def load_sqlite_page_candidates(page_ids: list[int]) -> pd.DataFrame:
    """Load article metadata for IDs returned by an independent retriever."""
    if not page_ids:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in page_ids)
    query = f"""
        SELECT
            {ARTICLE_COLUMNS},
            0.0 AS bm25_score,
            0 AS exact_title_match,
            0 AS lead_exact_match,
            0 AS exact_full_text_match
        FROM articles
        WHERE NOT articles.is_redirect
          AND articles.page_id IN ({placeholders})
    """
    database_uri = f"file:{SQLITE_SEARCH_PATH}?mode=ro&immutable=1"
    with sqlite3.connect(database_uri, uri=True) as connection:
        return pd.read_sql_query(query, connection, params=page_ids)


@lru_cache(maxsize=1)
def load_kalm_model():
    """Load KaLM only for the optional non-city query fallback."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "This query is not a recognized city alias and needs the optional "
            "semantic dependencies. Install with: pip install -e '.[semantic]'"
        ) from exc

    model = SentenceTransformer(
        KALM_MODEL_NAME,
        trust_remote_code=True,
        device="cpu",
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
            "python preprocessing/07_build_embedding_memmap.py --overwrite"
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
def query_embedding(location_name: str) -> np.ndarray | None:
    """Load or compute one normalized embedding for a search query."""
    query_embedding = precomputed_city_embedding(location_name)
    if query_embedding is not None:
        return query_embedding
    # A qualified place query still has an exact FTS/BM25 path. Do not make
    # it depend on downloading the optional semantic model.
    if "," in location_name:
        return None
    try:
        model = load_kalm_model()
        return model.encode(
            [location_name],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
    except (ImportError, OSError, RuntimeError):
        return None


@lru_cache(maxsize=16)
def kalm_scores(
    location_name: str,
    target_page_ids: tuple[int, ...],
) -> pd.Series:
    """Return cosine similarity for selected article page IDs."""
    embedding_matrix, embedding_page_ids = load_kalm_embeddings()
    if len(embedding_page_ids) == 0:
        return pd.Series(dtype="float32")

    query_vector = query_embedding(location_name)
    if query_vector is None:
        return pd.Series(dtype="float32")
    if query_vector.shape != (embedding_matrix.shape[1],):
        raise ValueError(
            "Query and article embedding dimensions do not match: "
            f"{query_vector.shape} and {embedding_matrix.shape}"
        )
    requested_page_ids = np.fromiter(target_page_ids, dtype=np.int64)
    rows = np.searchsorted(embedding_page_ids, requested_page_ids)
    valid = rows < len(embedding_page_ids)
    valid[valid] &= embedding_page_ids[rows[valid]] == requested_page_ids[valid]
    rows = rows[valid]
    selected_page_ids = requested_page_ids[valid]
    if len(rows) == 0:
        return pd.Series(dtype="float32")

    # Fancy indexing materializes only selected rows from the memory map. The
    # complete matrix remains on disk instead of being copied into process RAM.
    candidate_matrix = np.array(
        embedding_matrix[rows],
        dtype=np.float32,
        order="C",
        copy=True,
    )
    # Embeddings are normalized at generation time, so the dot product is
    # cosine similarity. Runtime search deliberately stays NumPy/CPU-only.
    selected_scores = candidate_matrix @ query_vector
    return pd.Series(
        dict(zip(selected_page_ids.tolist(), selected_scores.astype(float))),
        dtype="float32",
    )


@lru_cache(maxsize=16)
def dense_retrieve_page_ids(
    location_name: str,
    top_k: int,
) -> pd.DataFrame:
    """Retrieve the top articles by scoring the complete embedding matrix."""
    embedding_matrix, embedding_page_ids = load_kalm_embeddings()
    query_vector = query_embedding(location_name)
    if query_vector is None or len(embedding_page_ids) == 0:
        return pd.DataFrame(columns=["page_id", "dense_score"])
    if query_vector.shape != (embedding_matrix.shape[1],):
        raise ValueError(
            "Query and article embedding dimensions do not match: "
            f"{query_vector.shape} and {embedding_matrix.shape}"
        )

    scores = np.asarray(embedding_matrix @ query_vector, dtype=np.float32)
    limit = min(max(int(top_k), 0), len(scores))
    if limit == 0:
        return pd.DataFrame(columns=["page_id", "dense_score"])
    if limit == len(scores):
        candidate_rows = np.arange(len(scores))
    else:
        candidate_rows = np.argpartition(scores, -limit)[-limit:]
    candidate_rows = candidate_rows[np.argsort(scores[candidate_rows])[::-1]]
    return pd.DataFrame(
        {
            "page_id": embedding_page_ids[candidate_rows].astype(np.int64),
            "dense_score": scores[candidate_rows].astype(float),
        }
    )


def normalize_scores(values: pd.Series) -> pd.Series:
    """Min-max normalize one query's scores to [0, 1]."""
    minimum = values.min()
    maximum = values.max()
    if pd.isna(minimum) or maximum <= minimum:
        return pd.Series(0.0, index=values.index)
    return (values - minimum) / (maximum - minimum)


def reciprocal_rank_fusion(
    rankings: dict[str, pd.DataFrame],
    *,
    top_n: int | None = None,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Fuse independently ranked result frames with Reciprocal Rank Fusion."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    weights = weights or {}
    fused: dict[int, dict[str, object]] = {}
    for name, frame in rankings.items():
        if "page_id" not in frame:
            raise ValueError(f"Ranking {name!r} is missing page_id")
        weight = float(weights.get(name, 1.0))
        for rank, page_id in enumerate(
            frame.drop_duplicates("page_id")["page_id"].tolist(), start=1
        ):
            page_id = int(page_id)
            entry = fused.setdefault(
                page_id,
                {"page_id": page_id, "rrf_score": 0.0, "retrieval_sources": []},
            )
            entry["rrf_score"] += weight / (k + rank)
            entry["retrieval_sources"].append(name)
    result = pd.DataFrame(fused.values())
    if result.empty:
        return pd.DataFrame(columns=["page_id", "rrf_score", "retrieval_sources"])
    result["retrieval_sources"] = result["retrieval_sources"].map(
        lambda sources: ",".join(sources)
    )
    result = result.sort_values("rrf_score", ascending=False, kind="stable")
    return result.head(top_n).reset_index(drop=True) if top_n else result.reset_index(drop=True)


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


def article_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build modest quality signals from article shape and metadata."""
    first_length = pd.to_numeric(
        frame.get("first_paragraph_length", frame["first_paragraph"].str.len()),
        errors="coerce",
    ).fillna(0)
    full_length = pd.to_numeric(
        frame.get("full_text_length", first_length),
        errors="coerce",
    ).fillna(0)
    recognized_type = frame["entity_class"].ne("other").astype(float)
    metadata_fields = ["wikidata_id", "country", "year", "lat", "lon"]
    metadata_score = sum(
        frame[field].notna().astype(float)
        for field in metadata_fields
        if field in frame
    )
    stub_like = (first_length < 160) & (full_length < 800)
    substantive_lead = (first_length >= 240).astype(float)
    incidental_mention = (
        frame["exact_full_text_match"]
        & ~frame["exact_title_match"]
        & ~frame["lead_exact_match"]
    )

    frame["quality_score"] = (
        -5.0 * frame["is_list_article"].astype(float)
        -2.5 * stub_like.astype(float)
        -1.5 * incidental_mention.astype(float)
        +0.75 * recognized_type
        +0.2 * metadata_score
        +1.0 * substantive_lead
        -0.75 * (first_length < 80).astype(float)
    )
    return frame


# ── Retrieval and ranking stages ──────────────────────────────────────────────

def prepare_search_features(
    frame: pd.DataFrame,
    location_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Add shared text, quality, citation, and normalized score features."""
    frame = article_quality_features(place_occurrence_features(frame.copy(), location_name))
    frame["bm25_score"] = pd.to_numeric(frame["bm25_score"], errors="coerce").fillna(0.0)
    frame["exact_score"] = (
        10 * frame["exact_title_match"].astype(float)
        + 3 * frame["exact_full_text_match"].astype(float)
    )
    text_match = (
        frame["exact_title_match"]
        | frame["lead_exact_match"]
        | frame["exact_full_text_match"]
    )
    citation_counts = load_citation_counts(location_name)
    frame["citation_context_count"] = frame["page_id"].map(citation_counts).fillna(0).astype("int8")
    frame["citation_penalty"] = np.where(
        frame["exact_title_match"] | frame["lead_exact_match"],
        0.0,
        np.minimum(8.0, 4.0 * frame["citation_context_count"]),
    )
    frame["kalm_score"] = frame["page_id"].map(
        kalm_scores(location_name, tuple(frame.loc[text_match, "page_id"].tolist()))
    ).fillna(0.0)
    frame["dense_score"] = pd.to_numeric(
        frame.get("dense_score", np.nan), errors="coerce"
    )
    frame["bm25_score_normalized"] = 0.0
    frame["kalm_score_normalized"] = 0.0
    frame["dense_score_normalized"] = 0.0
    frame.loc[text_match, "bm25_score_normalized"] = normalize_scores(
        frame.loc[text_match, "bm25_score"]
    )
    frame.loc[text_match, "kalm_score_normalized"] = normalize_scores(
        frame.loc[text_match, "kalm_score"]
    )
    dense_match = frame["dense_score"].notna()
    frame.loc[dense_match, "dense_score_normalized"] = normalize_scores(
        frame.loc[dense_match, "dense_score"]
    )
    return frame, text_match


def retrieve_coordinate_candidates(
    frame: pd.DataFrame,
    user_lat: float,
    user_lon: float,
    radius_km: float,
) -> pd.DataFrame:
    """Retrieve nearby articles and attach their haversine distances."""
    coords = frame[["page_id", "lat", "lon"]].dropna(subset=["lat", "lon"])
    lat1 = math.radians(user_lat)
    lat2 = np.radians(coords["lat"].to_numpy(dtype=np.float64))
    dlat = lat2 - lat1
    dlambda = np.radians(coords["lon"].to_numpy(dtype=np.float64) - user_lon)
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2) ** 2
    distances = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_by_page_id = pd.Series(distances, index=coords["page_id"].to_numpy(), dtype="float64")
    frame = frame.copy()
    frame["distance_km"] = frame["page_id"].map(distance_by_page_id)
    candidates = frame[frame["distance_km"].notna() & (frame["distance_km"] <= radius_km)].copy()
    candidates["source"] = "coordinates"
    return candidates


def retrieve_bm25_candidates(
    frame: pd.DataFrame,
    text_match: pd.Series,
    nearby_ids: set,
    bm25_min_score: float,
) -> pd.DataFrame:
    """Retrieve text-only candidates from the BM25 branch."""
    candidates = frame[
        (frame["bm25_score"] >= bm25_min_score)
        & text_match
        & ~frame["page_id"].isin(nearby_ids)
        & frame["lat"].isna()
    ].copy()
    candidates["distance_km"] = np.nan
    candidates["source"] = "bm25"
    return candidates


def retrieve_dense_candidates(
    location_name: str,
    kalm_top_k: int,
) -> pd.DataFrame:
    """Retrieve top candidates by scoring the full article matrix."""
    ranking = dense_retrieve_page_ids(location_name, kalm_top_k)
    candidates = load_sqlite_page_candidates(ranking["page_id"].astype(int).tolist())
    if candidates.empty:
        return candidates
    candidates = candidates.merge(ranking, on="page_id", how="inner")
    candidates["distance_km"] = np.nan
    candidates["source"] = "dense"
    return candidates


def combine_retrieval_results(*candidate_frames: pd.DataFrame) -> pd.DataFrame:
    """Combine branches, preferring coordinate evidence on overlap."""
    return pd.concat(candidate_frames).drop_duplicates(subset="page_id", keep="first")


def score_candidates(frame: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    """Apply the current text-led ranking policy to retrieved candidates."""
    frame = frame.copy()
    editorial_importance = load_editorial_importance()
    frame["editorial_importance"] = frame["page_id"].map(editorial_importance).fillna(0.0)
    score = frame["exact_score"].astype(float) - frame["citation_penalty"]
    score += (
        5 * frame["bm25_score_normalized"]
        + 2 * frame["kalm_score_normalized"]
        + 2 * frame["dense_score_normalized"]
    )
    score += (
        8 * frame["place_title_mentions"]
        + 5 * frame["place_lead_mentions"]
        + 2 * np.log1p(frame["place_title_mentions"] + frame["place_lead_mentions"])
        + 4 * frame["place_position_score"]
    )
    score += (0.5 * (1 - frame["distance_km"] / max(radius_km, 1.0))).clip(lower=0).fillna(0)
    score += ((5 - frame["hop"]) * 2).fillna(0)
    score += frame["quality_score"] + 0.25 * frame["editorial_importance"]
    frame["score"] = score
    return frame


def finalize_results(frame: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep all mapped results and the best text-only results."""
    coordinate_results = frame[frame["distance_km"].notna()]
    text_results = frame[frame["distance_km"].isna()].nlargest(top_n, "score")
    results = pd.concat([coordinate_results, text_results.head(top_n)], ignore_index=True)
    return results.sort_values("score", ascending=False), {
        "total_found": len(frame),
        "from_coords": int((frame["source"] == "coordinates").sum()),
    }


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

    dense_index = retrieve_dense_candidates(location_name, kalm_top_k)
    dense_scores = (
        dense_index[["page_id", "dense_score"]]
        if not dense_index.empty
        else pd.DataFrame(columns=["page_id", "dense_score"])
    )
    combined_index = pd.concat([index, dense_index], ignore_index=True).drop_duplicates(
        subset="page_id", keep="first"
    )
    combined_index = combined_index.drop(columns=["source", "distance_km"], errors="ignore")
    combined_index = combined_index.drop(columns=["dense_score"], errors="ignore").merge(
        dense_scores, on="page_id", how="left"
    )
    features, text_match = prepare_search_features(combined_index, location_name)
    coordinate_candidates = retrieve_coordinate_candidates(
        features, user_lat, user_lon, radius_km
    )
    nearby_ids = set(coordinate_candidates["page_id"])
    bm25_candidates = retrieve_bm25_candidates(
        features, text_match, nearby_ids, bm25_min_score
    )
    dense_candidates = features[
        features["dense_score"].notna()
        & ~features["page_id"].isin(nearby_ids)
        & features["lat"].isna()
    ].copy()
    dense_candidates["distance_km"] = np.nan
    dense_candidates["source"] = "dense"
    combined = combine_retrieval_results(
        coordinate_candidates,
        bm25_candidates,
        dense_candidates,
    )
    scored = score_candidates(combined, radius_km)
    results, stats = finalize_results(scored, top_n)
    return {"results": results, "stats": stats}
