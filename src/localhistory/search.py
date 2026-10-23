"""Retrieval and ranking for local-history articles.

The live pipeline combines SQLite coordinate/BM25 retrieval with FAISS dense
retrieval, fuses those rankings, applies local-history relevance features, and
uses a cross-encoder to rerank the top text candidates.
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
    from .city_queries import precomputed_city_embedding
except ImportError:  # Support running search.py/profile_search.py as scripts.
    from localhistory.city_queries import precomputed_city_embedding

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
KALM_EMBEDDING_MATRIX_PATH = DATA_DIR / "kalm_embeddings.npy"
KALM_EMBEDDING_PAGE_IDS_PATH = DATA_DIR / "kalm_embedding_page_ids.npy"
FAISS_INDEX_PATH = DATA_DIR / "kalm_embeddings.faiss"
KALM_MODEL_NAME = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Ranking configuration. Keep retrieval limits separate from model settings so
# search quality/latency experiments only require changing this section.
RRF_K = 60
DEFAULT_DENSE_TOP_K = 200
RERANKER_TOP_K = 50
RERANKER_WEIGHT = 5.0
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
    articles.location,
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
    """Load text matches plus nearby candidates for later text filtering."""
    require_current_index(
        SQLITE_SEARCH_PATH,
        INDEX_PATH,
        "python preprocessing/05_build_sqlite_search.py --overwrite",
    )
    text_candidates = load_sqlite_text_candidates(location_name)
    text_candidates["query_text_match"] = True
    entity_candidates = load_sqlite_entity_candidates(location_name)
    entity_candidates["query_text_match"] = False
    nearby_candidates = load_sqlite_coordinate_candidates(
        user_lat,
        user_lon,
        radius_km,
    )
    nearby_candidates["query_text_match"] = False
    nearby_candidates["query_match_tier"] = 0
    all_candidates = pd.concat(
        [text_candidates, entity_candidates, nearby_candidates]
    )
    category_scores = all_candidates.groupby("page_id")["category_score"].max()
    candidates = all_candidates.drop_duplicates(
        subset="page_id",
        keep="first",
    )
    candidates["category_score"] = candidates["page_id"].map(category_scores).fillna(0.0)
    for column in (
        "is_list_article",
        "query_text_match",
        "exact_title_match",
        "lead_exact_match",
        "exact_full_text_match",
        "entity_associated",
    ):
        values = candidates.get(
            column,
            pd.Series(False, index=candidates.index, dtype=bool),
        )
        candidates[column] = values.fillna(False).astype(bool)
    candidates["category_score"] = pd.to_numeric(
        candidates.get(
            "category_score",
            pd.Series(0.0, index=candidates.index, dtype="float64"),
        ),
        errors="coerce",
    ).fillna(0.0)
    candidates["query_match_tier"] = (
        pd.to_numeric(candidates["query_match_tier"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
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


def quote_fts_phrase(value: str) -> str:
    return '"' + value.strip().lower().replace('"', '""') + '"'


def load_sqlite_text_candidates(location_name: str) -> pd.DataFrame:
    """Retrieve exact-phrase and unordered city/qualifier matches.

    City-only lexical retrieval is deliberately excluded because a bare city
    name can be ambiguous. Exact phrase matches receive the stronger tier, but
    unordered terms are always retained rather than used only as a sparse
    fallback.
    """
    database_uri = f"file:{SQLITE_SEARCH_PATH}?mode=ro&immutable=1"
    parts = [part.strip() for part in location_name.split(",") if part.strip()]
    city_phrase = quote_fts_phrase(parts[0])
    strict_phrase = quote_fts_phrase(" ".join(parts))
    qualified_terms = " AND ".join(quote_fts_phrase(part) for part in parts)

    def run_query(connection: sqlite3.Connection, phrase: str) -> pd.DataFrame:
        query = f"""
            SELECT
                {ARTICLE_COLUMNS},
                -bm25(article_fts, 1.0, 1.0, 1.0) AS bm25_score
            FROM article_fts
            JOIN articles ON articles.page_id = article_fts.rowid
            WHERE article_fts MATCH ?
              AND NOT articles.is_redirect
            ORDER BY articles.page_id
        """
        return pd.read_sql_query(query, connection, params=[phrase])

    with sqlite3.connect(database_uri, uri=True) as connection:
        strict = run_query(connection, strict_phrase)
        strict["query_match_tier"] = 2
        relaxed = run_query(connection, qualified_terms)
        relaxed["query_match_tier"] = 1
        candidates = pd.concat([strict, relaxed], ignore_index=True).drop_duplicates(
            subset="page_id", keep="first"
        )
        candidates["entity_associated"] = False
        candidates["category_score"] = 0.0
        for fts_column, result_column in (
            ("title", "exact_title_match"),
            ("first_paragraph", "lead_exact_match"),
            ("full_text", "exact_full_text_match"),
        ):
            matches = connection.execute(
                "SELECT rowid FROM article_fts WHERE article_fts MATCH ?",
                [f"{fts_column} : {city_phrase}"],
            )
            page_ids = {page_id for (page_id,) in matches}
            candidates[result_column] = candidates["page_id"].isin(page_ids)

    return candidates


def load_sqlite_entity_candidates(location_name: str) -> pd.DataFrame:
    """Retrieve articles whose structured location metadata names the city.

    This uses geographic metadata and article categories. Link relationships
    are not present in the current extraction artifacts and can be added as a
    further branch when raw wiki links are retained.
    """
    parts = [part.strip() for part in location_name.split(",") if part.strip()]
    if not parts:
        return pd.DataFrame()
    city = parts[0].casefold()
    location_query = f"""
        SELECT
            {ARTICLE_COLUMNS},
            1.0 AS bm25_score,
            1 AS entity_associated,
            1 AS query_match_tier,
            0 AS exact_title_match,
            0 AS lead_exact_match,
            0 AS exact_full_text_match
        FROM articles
        WHERE NOT articles.is_redirect
          AND (
                lower(COALESCE(articles.location, '')) = ?
             OR lower(COALESCE(articles.location, '')) LIKE ?
             OR lower(COALESCE(articles.location, '')) LIKE ?
             OR lower(COALESCE(articles.location, '')) LIKE ?
          )
    """
    category_query = f"""
        SELECT
            {ARTICLE_COLUMNS},
            article_categories.category AS matched_category,
            1.0 AS bm25_score,
            1 AS entity_associated,
            1 AS query_match_tier,
            0 AS exact_title_match,
            0 AS lead_exact_match,
            0 AS exact_full_text_match
        FROM articles
        JOIN article_categories
          ON article_categories.page_id = articles.page_id
        WHERE NOT articles.is_redirect
          AND (CATEGORY_MATCH_CONDITION)
    """
    qualifiers = [part.casefold() for part in parts[1:]]
    terms = [city, *qualifiers]
    category_conditions = []
    category_params = []
    for term in terms:
        category_conditions.append(
            "("
            "lower(replace(replace(replace(article_categories.category, '_', ' '), '(', ' '), ')', ' ')) = ? "
            "OR lower(replace(replace(replace(article_categories.category, '_', ' '), '(', ' '), ')', ' ')) LIKE ? "
            "OR lower(replace(replace(replace(article_categories.category, '_', ' '), '(', ' '), ')', ' ')) LIKE ? "
            "OR lower(replace(replace(replace(article_categories.category, '_', ' '), '(', ' '), ')', ' ')) LIKE ? "
            "OR lower(replace(replace(replace(article_categories.category, '_', ' '), '(', ' '), ')', ' ')) LIKE ?"
            ")"
        )
        category_params.extend(
            [
                term,
                f"{term} %",
                f"% {term}",
                f"% {term} %",
                f"% {term},%",
            ]
        )
    category_query = category_query.replace(
        "CATEGORY_MATCH_CONDITION",
        " OR ".join(category_conditions),
    )
    location_patterns = [city, f"{city} %", f"% {city}", f"% {city},%"]
    database_uri = f"file:{SQLITE_SEARCH_PATH}?mode=ro&immutable=1"
    with sqlite3.connect(database_uri, uri=True) as connection:
        frames = [
            pd.read_sql_query(
                location_query,
                connection,
                params=location_patterns,
            )
        ]
        has_categories = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'article_categories'
            """
        ).fetchone()
        if has_categories:
            frames.append(
                pd.read_sql_query(
                    category_query,
                    connection,
                    params=category_params,
                )
            )
    candidates = pd.concat(frames, ignore_index=True)
    candidates["category_score"] = 0.0
    if "matched_category" in candidates:
        history_words = re.compile(
            r"(?:history|historical|medieval|revolution|war|military|"
            r"political|economic|social|cultural|architecture|timeline|"
            r"archaeolog|historic)",
            flags=re.IGNORECASE,
        )
        city_pattern = re.compile(
            rf"(?:^|[ _(),-]){re.escape(city)}(?:$|[ _(),-])",
            flags=re.IGNORECASE,
        )
        qualifier_patterns = [
            re.compile(
                rf"(?:^|[ _(),-]){re.escape(term)}(?:$|[ _(),-])",
                flags=re.IGNORECASE,
            )
            for term in qualifiers
        ]

        def category_score(category: object) -> float:
            value = str(category or "")
            city_hit = bool(city_pattern.search(value))
            qualifier_hit = any(pattern.search(value) for pattern in qualifier_patterns)
            historical = bool(history_words.search(value))
            if city_hit and historical:
                return 1.0
            if city_hit:
                return 0.75
            if qualifier_hit and historical:
                return 0.45
            if qualifier_hit:
                return 0.20
            return 0.0

        candidates["category_score"] = candidates["matched_category"].map(
            category_score
        )
    return candidates.sort_values(
        "category_score", ascending=False, kind="stable"
    ).drop_duplicates(subset="page_id", keep="first")


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
def load_cross_encoder():
    """Load the second-stage cross-encoder once per API process."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "The cross-encoder requires sentence-transformers. Install the "
            "optional semantic dependencies with: pip install -e '.[semantic]'"
        ) from exc

    return CrossEncoder(
        RERANKER_MODEL_NAME,
        device=os.getenv("LOCAL_HISTORY_RERANKER_DEVICE", "cpu"),
        max_length=512,
    )


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


@lru_cache(maxsize=1)
def load_faiss_index():
    """Load the persisted exact inner-product index for dense retrieval."""
    if os.getenv("LOCAL_HISTORY_USE_FAISS", "1").casefold() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    if not FAISS_INDEX_PATH.exists():
        return None
    try:
        import faiss
    except ImportError:
        return None
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    embedding_matrix, _ = load_kalm_embeddings()
    if index.d != embedding_matrix.shape[1]:
        raise ValueError(
            "FAISS and embedding dimensions do not match: "
            f"{index.d} and {embedding_matrix.shape[1]}"
        )
    return index


@lru_cache(maxsize=16)
def query_embedding(location_name: str) -> np.ndarray | None:
    """Load or compute one normalized embedding for a search query."""
    query_embedding = precomputed_city_embedding(location_name)
    if query_embedding is not None:
        return query_embedding
    # A place query still has an exact FTS/BM25 path. Do not make
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
def dense_scores(
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
    """Retrieve top articles with FAISS, falling back to NumPy."""
    query_vector = query_embedding(location_name)
    if query_vector is None:
        return pd.DataFrame(columns=["page_id", "dense_score"])
    limit = max(int(top_k), 0)
    if limit == 0:
        return pd.DataFrame(columns=["page_id", "dense_score"])

    index = load_faiss_index()
    if index is not None:
        scores, page_ids = index.search(
            np.asarray(query_vector, dtype=np.float32).reshape(1, -1),
            min(limit, index.ntotal),
        )
        valid = page_ids[0] >= 0
        return pd.DataFrame(
            {
                "page_id": page_ids[0][valid].astype(np.int64),
                "dense_score": scores[0][valid].astype(float),
            }
        )

    # Compatibility fallback for environments without FAISS or its index file.
    embedding_matrix, embedding_page_ids = load_kalm_embeddings()
    if query_vector.shape != (embedding_matrix.shape[1],):
        raise ValueError(
            "Query and article embedding dimensions do not match: "
            f"{query_vector.shape} and {embedding_matrix.shape}"
        )
    scores = np.asarray(embedding_matrix @ query_vector, dtype=np.float32)
    limit = min(limit, len(scores))
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
    frame["quality_score"] = (
        -5.0 * frame["is_list_article"].astype(float)
        -2.5 * stub_like.astype(float)
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
    """Add quality and normalized BM25/dense score features."""
    for column in (
        "is_list_article",
        "query_text_match",
        "exact_title_match",
        "lead_exact_match",
        "exact_full_text_match",
    ):
        frame[column] = frame[column].fillna(False).astype(bool)
    frame = article_quality_features(frame.copy())
    frame["bm25_score"] = pd.to_numeric(frame["bm25_score"], errors="coerce").fillna(0.0)
    text_match = frame["query_text_match"]
    frame["kalm_score"] = frame["page_id"].map(
        dense_scores(location_name, tuple(frame.loc[text_match, "page_id"].tolist()))
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
    dense_top_k: int,
) -> pd.DataFrame:
    """Retrieve top candidates from the FAISS dense index."""
    ranking = dense_retrieve_page_ids(location_name, dense_top_k)
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


def fuse_text_retrieval_ranks(
    bm25_candidates: pd.DataFrame,
    dense_candidates: pd.DataFrame,
    category_candidates: pd.DataFrame,
    *,
    k: int = 60,
) -> pd.DataFrame:
    """Fuse BM25, dense, and structured category rankings with RRF."""
    strict_bm25 = bm25_candidates[bm25_candidates["query_match_tier"] == 2]
    relaxed_bm25 = bm25_candidates[bm25_candidates["query_match_tier"] == 1]
    rankings = {
        "bm25_strict": strict_bm25.sort_values(
            "bm25_score", ascending=False, kind="stable"
        )[["page_id"]],
        "bm25_relaxed": relaxed_bm25.sort_values(
            "bm25_score", ascending=False, kind="stable"
        )[["page_id"]],
        "kalm": dense_candidates.sort_values(
            "dense_score", ascending=False, kind="stable"
        )[["page_id"]],
        "category": category_candidates.sort_values(
            ["category_score", "quality_score"],
            ascending=False,
            kind="stable",
        )[["page_id"]],
    }
    return reciprocal_rank_fusion(
        rankings,
        k=k,
        weights={
            "bm25_strict": 4.0,
            "bm25_relaxed": 1.0,
            "kalm": 1.0,
            "category": 5.0,
        },
    )


def score_candidates(frame: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    """Apply the current text-led ranking policy to retrieved candidates."""
    frame = frame.copy()
    # RRF scores are small (roughly 0.01–0.03 for k=60), so scale them into
    # the same practical range as the other ranking signals. The rank fusion
    # itself remains independent of raw BM25/cosine score magnitudes.
    score = 100 * pd.to_numeric(
        frame["rrf_score"], errors="coerce"
    ).fillna(0.0)
    distance_km = pd.to_numeric(frame["distance_km"], errors="coerce")
    hop = pd.to_numeric(frame["hop"], errors="coerce")
    quality = pd.to_numeric(frame["quality_score"], errors="coerce").fillna(0.0)
    category_score = pd.to_numeric(
        frame.get(
            "category_score",
            pd.Series(0.0, index=frame.index, dtype="float64"),
        ),
        errors="coerce",
    ).fillna(0.0)
    score += (0.5 * (1 - distance_km / max(radius_km, 1.0))).clip(lower=0).fillna(0)
    score += ((5 - hop) * 2).fillna(0)
    score += quality
    score += 4.0 * category_score
    frame["score"] = pd.to_numeric(score, errors="coerce").fillna(0.0).astype(float)
    return frame


def rerank_with_cross_encoder(
    frame: pd.DataFrame,
    location_name: str,
    *,
    top_k: int = RERANKER_TOP_K,
) -> pd.DataFrame:
    """Rerank the best text candidates after RRF with a cross-encoder."""
    frame = frame.copy()
    frame["cross_encoder_score"] = np.nan
    eligible = frame[frame["distance_km"].isna()].nlargest(top_k, "score")
    if eligible.empty:
        return frame

    pairs = [
        (
            location_name,
            f"{row.title}. {row.first_paragraph}",
        )
        for row in eligible.itertuples()
    ]
    reranker_scores = np.asarray(
        load_cross_encoder().predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        ),
        dtype=float,
    ).reshape(-1)
    score_by_page_id = pd.Series(
        reranker_scores,
        index=eligible["page_id"].to_numpy(),
        dtype="float64",
    )
    frame["cross_encoder_score"] = frame["page_id"].map(score_by_page_id)
    reranked = frame["cross_encoder_score"].notna()
    frame.loc[reranked, "score"] += RERANKER_WEIGHT * normalize_scores(
        frame.loc[reranked, "cross_encoder_score"]
    )
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
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    index: pd.DataFrame = None,
) -> dict:
    if index is None:
        index = load_candidate_index(
            location_name,
            user_lat,
            user_lon,
            radius_km,
        )

    dense_index = retrieve_dense_candidates(location_name, dense_top_k)
    dense_scores = (
        dense_index[["page_id", "dense_score"]]
        if not dense_index.empty
        else pd.DataFrame(columns=["page_id", "dense_score"])
    )
    combined_index = pd.concat([index, dense_index], ignore_index=True).drop_duplicates(
        subset="page_id", keep="first"
    )
    combined_index["query_text_match"] = combined_index["query_text_match"].fillna(False).astype(bool)
    combined_index["entity_associated"] = (
        combined_index.get(
            "entity_associated",
            pd.Series(False, index=combined_index.index, dtype=bool),
        )
        .fillna(False)
        .astype(bool)
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
    category_candidates = features[
        (features["category_score"] > 0)
        & ~features["page_id"].isin(nearby_ids)
        & features["lat"].isna()
    ].copy()
    category_candidates["distance_km"] = np.nan
    category_candidates["source"] = "category"
    rrf_scores = fuse_text_retrieval_ranks(
        bm25_candidates,
        dense_candidates,
        category_candidates,
        k=RRF_K,
    )
    combined = combine_retrieval_results(
        coordinate_candidates,
        bm25_candidates,
        dense_candidates,
        category_candidates,
    )
    combined = combined.merge(
        rrf_scores[["page_id", "rrf_score"]],
        on="page_id",
        how="left",
    )
    scored = score_candidates(combined, radius_km)
    scored = rerank_with_cross_encoder(scored, location_name)
    results, stats = finalize_results(scored, top_n)
    return {"results": results, "stats": stats}
