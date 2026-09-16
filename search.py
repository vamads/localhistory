"""
search.py

Hybrid local history search combining:
  1. Coordinate proximity   — articles within radius_km of user location
  2. BM25 retrieval          — fallback for articles without coordinates
  3. Scoring & ranking      — BM25, distance, entity class, hop, date

Usage:
    python search.py
"""

import ast
import json
import math
import os
import re
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import bm25s
import torch

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
BM25_INDEX_PATH = DATA_DIR / "bm25_index"
BM25_DOC_IDS_PATH = DATA_DIR / "bm25_doc_ids.json"
KALM_EMBEDDINGS_DIR = DATA_DIR / "kalm_first_paragraph_embeddings"
KALM_MODEL_NAME = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"


def load_candidate_index(location_name: str) -> pd.DataFrame:
    """Load only articles containing the queried city in searchable text."""
    city_name = location_name.split(",", 1)[0].strip().lower()
    query = """
        SELECT *
        FROM read_parquet(?)
        WHERE NOT is_redirect
          AND (
              contains(lower(coalesce(title, '')), ?)
              OR contains(lower(coalesce(first_paragraph, '')), ?)
              OR contains(lower(coalesce(full_text, '')), ?)
          )
    """
    with duckdb.connect() as connection:
        return connection.execute(
            query,
            [str(INDEX_PATH), city_name, city_name, city_name],
        ).fetchdf()


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

# ── Entity type sets ──────────────────────────────────────────────────────────

PERSON_TYPES = {"human", "person", "fictional human", "fictional character"}

EVENT_TYPES = {
    "battle",
    "war",
    "revolution",
    "historical event",
    "event",
    "armed conflict",
    "conflict",
    "election",
    "treaty",
    "massacre",
    "siege",
    "incident",
    "protest",
    "rebellion",
    "uprising",
    "ethnic riot",
    "riot",
    "race riot",
    "pogrom",
    "labor dispute",
    "strike action",
    "march",
    "demonstration",
    "coup",
    "assassination",
    "execution",
    "trial",
    "disaster",
    "fire",
    "flood",
    "earthquake",
    "famine",
    "epidemic",
    "pandemic",
    "expedition",
    "voyage",
}

PLACE_TYPES = {
    "city",
    "municipality",
    "town",
    "village",
    "country",
    "state",
    "region",
    "archaeological site",
    "building",
    "church",
    "castle",
    "fort",
    "monument",
    "museum",
    "archaeological museum",
    "art museum",
    "history museum",
    "natural history museum",
    "library",
    "archive",
    "hospital",
    "psychiatric hospital",
    "factory",
    "assembly plant",
    "airport",
    "airfield",
    "lake",
    "reservoir",
    "island",
    "park",
    "synagogue",
    "mosque",
    "temple",
    "tekke",
    "university",
    "neighborhood",
    "historic district",
    "unincorporated community",
    "house",
    "duplex",
    "apartment building",
    "residential building",
    "public housing",
    "housing project",
    "estate",
    "manor",
    "garden",
    "national historic landmark",
    "heritage site",
    "listed building",
    "tower",
    "palace",
    "prison",
    "courthouse",
    "school",
    "college",
    "stadium",
    "arena",
    "theater",
    "theatre",
    "opera house",
    "cemetery",
    "battlefield",
    "memorial",
    "plaza",
    "square",
    "harbor",
    "port",
    "canal",
    "railway station",
    "train station",
    "road",
    "street",
    "bridge",
    "tunnel",
    "dam",
}

WORK_TYPES = {
    "book",
    "novel",
    "film",
    "painting",
    "newspaper",
    "journal",
    "magazine",
    "song",
    "album",
    "artwork",
    "document",
    "periodical",
    "underground press",
    "newsletter",
    "art project",
    "art installation",
    "public art",
    "mural",
    "sculpture",
    "photograph",
    "documentary",
    "television series",
    "radio program",
    "poem",
    "play",
    "opera",
    "musical",
    "comic book",
}

ORG_TYPES = {
    "organization",
    "nonprofit organization",
    "association",
    "political party",
    "trade union",
    "community organization",
    "religious organization",
    "diaspora organization",
    "episcopate",
    "diocese",
    "company",
    "institution",
    "government agency",
    "military unit",
    "regiment",
    "brigade",
    "society",
    "club",
    "fraternity",
    "sorority",
    "guild",
    "corporation",
    "foundation",
    "institute",
    "think tank",
    "newspaper publisher",
    "record label",
    "studio",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_instance_of(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    return []


def get_entity_class(instance_of, title="", first_para="") -> str:
    types = parse_instance_of(instance_of)
    types_lower = {t.lower().strip() for t in types} if types else set()

    if types_lower & PERSON_TYPES:
        return "person"
    if types_lower & EVENT_TYPES:
        return "event"
    if types_lower & PLACE_TYPES:
        return "place"
    if types_lower & WORK_TYPES:
        return "work"
    if types_lower & ORG_TYPES:
        return "organization"

    # Text-based fallback for empty or unmatched instance_of
    text = (title + " " + (first_para or "")).lower()
    if any(
        w in text
        for w in [
            "riot",
            "battle",
            "massacre",
            "siege",
            "uprising",
            "revolution",
            "rebellion",
            "march",
            "strike",
        ]
    ):
        return "event"
    if any(
        w in text
        for w in [
            "museum",
            "library",
            "building",
            "house",
            "park",
            "church",
            "hospital",
            "school",
            "neighborhood",
        ]
    ):
        return "place"
    if any(
        w in text
        for w in [
            "organization",
            "society",
            "movement",
            "association",
            "party",
            "union",
            "company",
            "corporation",
        ]
    ):
        return "organization"
    if any(
        w in text
        for w in [
            "newspaper",
            "magazine",
            "journal",
            "album",
            "novel",
            "film",
            "painting",
            "sculpture",
        ]
    ):
        return "work"
    return "other"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@lru_cache(maxsize=1)
def load_bm25_index():
    """Load the persisted BM25 index and its page-id mapping."""
    retriever = bm25s.BM25.load(
        BM25_INDEX_PATH,
        load_corpus=False,
        mmap=True,
    )
    with BM25_DOC_IDS_PATH.open() as f:
        page_ids = json.load(f)
    return retriever, page_ids


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
    """Load every stored embedding once and keep the matrix on the search device."""
    embedding_files = sorted(KALM_EMBEDDINGS_DIR.glob("embeddings_*.parquet"))
    if not embedding_files:
        return torch.empty((0, 0), device=kalm_device()), {}

    chunks = [pd.read_parquet(path, columns=["page_id", "embedding"])
              for path in embedding_files]
    embeddings = pd.concat(chunks, ignore_index=True)
    matrix = np.vstack(embeddings["embedding"].to_numpy()).astype(np.float32)
    page_ids = embeddings["page_id"].tolist()
    page_to_row = {page_id: row for row, page_id in enumerate(page_ids)}
    return torch.from_numpy(matrix).to(kalm_device()), page_to_row


@lru_cache(maxsize=16)
def kalm_scores(location_name: str, target_page_ids=None) -> pd.Series:
    """Return cosine similarity to each stored first-paragraph embedding."""
    embedding_matrix, page_to_row = load_kalm_embeddings()
    if not page_to_row:
        return pd.Series(dtype="float32")

    model = load_kalm_model()
    query_embedding = model.encode(
        [location_name],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)
    query = torch.from_numpy(query_embedding).to(embedding_matrix.device)

    if target_page_ids is None:
        rows = list(range(len(page_to_row)))
        selected_page_ids = list(page_to_row)
    else:
        selected_page_ids = [page_id for page_id in target_page_ids if page_id in page_to_row]
        rows = [page_to_row[page_id] for page_id in selected_page_ids]
    if not rows:
        return pd.Series(dtype="float32")

    # Score only the candidate rows. The previous implementation multiplied
    # the query by the entire embedding matrix before discarding almost all
    # scores, which is particularly expensive on MPS for a large corpus.
    selected_scores = (embedding_matrix[rows] @ query).detach().cpu().numpy()
    return pd.Series(
        dict(zip(selected_page_ids, selected_scores.astype(float))),
        dtype="float32",
    )


@lru_cache(maxsize=1)
def load_coordinate_index() -> pd.DataFrame:
    """Load coordinate columns once for vectorized radius filtering."""
    return pd.read_parquet(INDEX_PATH, columns=["page_id", "lat", "lon"])


def normalize_scores(values: pd.Series) -> pd.Series:
    """Min-max normalize one query's scores to [0, 1]."""
    minimum = values.min()
    maximum = values.max()
    if pd.isna(minimum) or maximum <= minimum:
        return pd.Series(0.0, index=values.index)
    return (values - minimum) / (maximum - minimum)


def city_core(location_name: str) -> str:
    """Use the city portion before an optional state/country qualifier."""
    return location_name.split(",", 1)[0].strip()


CITATION_MARKERS = (
    r"\b(?:press|publisher|publishing|university press|journal|"
    r"vol\.?|volume|pp?\.?|pages|isbn|doi|retrieved|accessed)\b"
)


def citation_context_count(text: str, location_pattern) -> int:
    """Count city mentions occurring in citation-like sentence contexts."""
    if text is None or pd.isna(text):
        text = ""
    citation_count = 0
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(text))

    for sentence in sentences:
        if not location_pattern.search(sentence):
            continue

        marker_count = len(re.findall(CITATION_MARKERS, sentence, re.IGNORECASE))
        has_year = bool(re.search(r"\b(?:18|19|20)\d{2}\b", sentence))
        has_bibliographic_shape = bool(
            re.search(r":[^.!?]{0,120},\s*(?:18|19|20)\d{2}\b", sentence)
        )

        # Require multiple signals unless the sentence has the characteristic
        # publisher/location/year shape, reducing false penalties in prose.
        if (marker_count >= 2 and has_year) or has_bibliographic_shape:
            citation_count += 1

    return citation_count


# ── Main search ───────────────────────────────────────────────────────────────


def search_local_history(
    location_name: str,
    user_lat: float,
    user_lon: float,
    radius_km: float = 50.0,
    top_n: int = 30,
    bm25_min_score: float = 1.0,
    kalm_top_k: int = 200,
    worst_n: int = 10,
    fallback_top_n: int = 15,
    index: pd.DataFrame = None,
) -> dict:
    if index is None:
        index = load_candidate_index(location_name)

    df = index[~index["is_redirect"]].copy()

    # Reuse the precomputed classification. Reclassifying every article on
    # every query is an expensive full-corpus row-wise operation. Older
    # indexes without this column still get the previous fallback behavior.
    if "entity_class" not in df.columns:
        df["entity_class"] = df.apply(
            lambda r: get_entity_class(
                r["instance_of"], r["title"], r["first_paragraph"]
            ),
            axis=1,
        )

    bm25_retriever, bm25_page_ids = load_bm25_index()
    query_tokens = bm25s.tokenize([location_name])
    query_terms = list(query_tokens.vocab.keys())
    bm25_scores = (
        bm25_retriever.get_scores(query_terms)
        if query_terms
        else np.zeros(len(bm25_page_ids), dtype=np.float32)
    )

    # Compute corpus-level BM25 scores without sorting every document. We
    # later retain these scores only for exact city-phrase candidates.
    score_by_page_id = dict(
        zip(bm25_page_ids, np.asarray(bm25_scores).astype(float))
    )
    df["bm25_score"] = df["page_id"].map(score_by_page_id).fillna(0.0)

    # Exact phrase evidence. A qualifier such as ", MI" or ", Michigan" is
    # ignored for matching, so both queries use the core phrase "Ann Arbor".
    city_name = city_core(location_name)
    location_pattern = re.compile(
        rf"(?<!\w){re.escape(city_name)}(?!\w)",
        re.IGNORECASE,
    )
    df["exact_title_match"] = df["title"].fillna("").str.contains(
        location_pattern, na=False
    )
    full_text = df["full_text"].fillna("")
    # The city name is a literal string, so avoid running the regex engine over
    # every long article. Recheck only literal matches with the boundary-aware
    # regex to preserve the original matching semantics.
    literal_full_text_match = full_text.str.contains(
        city_name, case=False, regex=False, na=False
    )
    df["exact_full_text_match"] = False
    literal_matches = literal_full_text_match[literal_full_text_match].index
    df.loc[literal_matches, "exact_full_text_match"] = full_text.loc[
        literal_matches
    ].map(lambda text: bool(location_pattern.search(text)))
    df["exact_score"] = (
        10 * df["exact_title_match"].astype(float)
        + 3 * df["exact_full_text_match"].astype(float)
    )
    df["lead_exact_match"] = df["first_paragraph"].fillna("").str.contains(
        location_pattern, na=False
    )
    exact_city_match = df["exact_title_match"] | df["exact_full_text_match"]
    df["citation_context_count"] = 0
    df.loc[exact_city_match, "citation_context_count"] = df.loc[
        exact_city_match, "full_text"
    ].apply(lambda text: citation_context_count(text, location_pattern))
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
        tuple(df.loc[exact_city_match, "page_id"].tolist()),
    )
    df["kalm_score"] = df["page_id"].map(kalm_by_page_id).fillna(0.0)
    df["bm25_score_normalized"] = normalize_scores(df["bm25_score"])
    df["kalm_score_normalized"] = normalize_scores(df["kalm_score"])

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
    exact_city_match = df["exact_title_match"] | df["exact_full_text_match"]
    bm25_matches = df[
        (df["bm25_score"] >= bm25_min_score)
        & exact_city_match
        & ~df["page_id"].isin(nearby_ids)
        & df["lat"].isna()
    ].copy()
    bm25_matches["distance_km"] = np.nan
    bm25_matches["source"] = "bm25"

    # Add exact-title and semantic candidates that BM25 may miss.
    ranked_kalm = df[exact_city_match].nlargest(kalm_top_k, "kalm_score")
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

    # Composite score
    def composite_score(row):
        s = 0.0
        s += row.get("exact_score", 0)
        s -= row.get("citation_penalty", 0)
        s += 5 * row.get("bm25_score_normalized", 0)
        s += 2 * row.get("kalm_score_normalized", 0)
        dist = row.get("distance_km")
        if pd.notna(dist) and dist is not None:
            s += max(0, 20 - dist * 0.4)
        hop = row.get("hop")
        if pd.notna(hop):
            s += (5 - hop) * 2
        s += {
            "event": 4,
            "place": 3,
            "person": 2,
            "work": 1,
            "organization": 2,
            "other": 0,
        }.get(row.get("entity_class", "other"), 0)
        if pd.notna(row.get("year")):
            s += 2
        if row.get("is_list_article", False):
            s -= 5
        return s

    combined["score"] = combined.apply(composite_score, axis=1)
    combined["coordinate_priority"] = (
        combined["source"] == "coordinates"
    ).astype(int)
    results = combined.sort_values(
        ["coordinate_priority", "score"],
        ascending=[False, False],
    ).head(top_n)
    worst_matches = combined.sort_values("score", ascending=True).head(worst_n)
    fallback_matches = (
        combined[combined["source"] != "coordinates"]
        .sort_values("score", ascending=False)
        .head(fallback_top_n)
    )

    # Timeline
    def century_label(year):
        if pd.isna(year) or year is None:
            return None
        y = int(year)
        if y < 0:
            c = abs(y) // 100 + 1
            sfx = {1: "st", 2: "nd", 3: "rd"}.get(
                c % 10 if c % 100 not in [11, 12, 13] else 0, "th"
            )
            return f"{c}{sfx} century BCE"
        c = y // 100 + 1
        sfx = {1: "st", 2: "nd", 3: "rd"}.get(
            c % 10 if c % 100 not in [11, 12, 13] else 0, "th"
        )
        return f"{c}{sfx} century"

    datable = results[results["year"].notna()].copy()
    datable["century"] = datable["year"].apply(century_label)
    by_century = defaultdict(list)
    for _, row in datable.sort_values("year").iterrows():
        by_century[row["century"]].append(row["title"])

    by_class = defaultdict(list)
    for _, row in results.iterrows():
        by_class[row["entity_class"]].append(row["title"])

    surprising = results[
        (results["entity_class"].isin({"work", "event"}))
        & (results["distance_km"].notna())
        & (results["distance_km"] < radius_km / 2)
    ].head(5)

    return {
        "results": results,
        "worst_matches": worst_matches,
        "fallback_matches": fallback_matches,
        "by_century": dict(by_century),
        "by_class": dict(by_class),
        "surprising": surprising,
        "stats": {
            "total_found": len(combined),
            "from_coords": (combined["source"] == "coordinates").sum(),
            "from_bm25": (combined["source"] == "bm25").sum(),
            "from_exact_or_kalm": (combined["source"] == "exact_or_kalm").sum(),
            "citation_context_matches": (
                combined["citation_context_count"] > 0
            ).sum(),
            "bm25_min_score": bm25_min_score,
            "bm25_matches_before_coordinate_filter": (
                (df["bm25_score"] >= bm25_min_score)
                & exact_city_match
                & df["lat"].isna()
            ).sum(),
            "with_dates": results["year"].notna().sum(),
        },
    }


# ── Print results ─────────────────────────────────────────────────────────────


def print_results(output: dict, location_name: str):
    stats = output["stats"]
    results = output["results"]

    print(f"\n{'='*65}")
    print(f"Local history near {location_name}")
    print(f"{'='*65}")
    print(
        f"Found {stats['total_found']} articles "
        f"({stats['from_coords']} by coordinates, "
        f"{stats['from_bm25']} by BM25 fallback, "
        f"{stats['from_exact_or_kalm']} exact/KaLM fallback)"
    )
    print(
        f"BM25 threshold: {stats['bm25_min_score']} "
        f"({stats['bm25_matches_before_coordinate_filter']} no-coordinate matches)"
    )
    print(
        f"Candidates with citation-like city context: "
        f"{stats['citation_context_matches']}"
    )
    print(f"{stats['with_dates']} have dates for timeline\n")

    print("── Top BM25/KaLM fallback matches ──────────────────────────")
    for _, row in output["fallback_matches"].iterrows():
        print(
            f"  [{row['source']:<14}] {row['title']} "
            f"score={row['score']:.2f} "
            f"(exact={row.get('exact_score', 0):.0f}, "
            f"citation_penalty={row.get('citation_penalty', 0):.0f}, "
            f"bm25={row.get('bm25_score', 0):.2f}, "
            f"kalm={row.get('kalm_score', 0):.3f})"
        )

    print("── Worst candidate matches ──────────────────────────────────")
    for _, row in output["worst_matches"].iterrows():
        print(
            f"  [{row['source']:<14}] {row['title']} "
            f"score={row['score']:.2f} "
            f"(exact={row.get('exact_score', 0):.0f}, "
            f"citation_penalty={row.get('citation_penalty', 0):.0f}, "
            f"bm25={row.get('bm25_score', 0):.2f}, "
            f"kalm={row.get('kalm_score', 0):.3f})"
        )

    print("── Top results ──────────────────────────────────────────────")
    for _, row in results.head(20).iterrows():
        dist = (
            f"{row['distance_km']:.1f}km"
            if pd.notna(row.get("distance_km"))
            else "BM25 match"
        )
        year = f" [{int(row['year'])}]" if pd.notna(row.get("year")) else ""
        print(
            f"  [{row['entity_class']:<12}] {row['title']}{year}  ({dist})  score={row['score']:.1f}"
        )

    if output["by_century"]:
        print("\n── Timeline ─────────────────────────────────────────────────")
        for century, titles in sorted(output["by_century"].items()):
            print(f"  {century}:")
            for t in titles[:3]:
                print(f"    • {t}")

    print("\n── By type ──────────────────────────────────────────────────")
    for cls, titles in output["by_class"].items():
        print(f"  {cls} ({len(titles)}): {', '.join(titles[:3])}")

    if len(output["surprising"]) > 0:
        print("\n── Surprising connections ───────────────────────────────────")
        for _, row in output["surprising"].iterrows():
            print(f"  {row['title']} [{row['entity_class']}]")


if __name__ == "__main__":
    print("Loading index ...")
    index = load_candidate_index("Ann Arbor")
    print(f"  {len(index):,} articles loaded")

    for location, lat, lon in [
        ("Ann Arbor", 42.2808, -83.7430),
        ("Detroit", 42.3314, -83.0458),
    ]:
        output = search_local_history(
            location_name=location,
            user_lat=lat,
            user_lon=lon,
            radius_km=50,
            top_n=30,
            index=index,
        )
        print_results(output, location)
        print()
