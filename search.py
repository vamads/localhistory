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
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import bm25s

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


# ── Main search ───────────────────────────────────────────────────────────────


def search_local_history(
    location_name: str,
    user_lat: float,
    user_lon: float,
    radius_km: float = 50.0,
    top_n: int = 30,
    index: pd.DataFrame = None,
) -> dict:
    if index is None:
        index = pd.read_parquet(INDEX_PATH)

    df = index[~index["is_redirect"]].copy()

    # Re-apply entity classification with updated type sets
    # (overrides whatever was saved in the index)
    df["entity_class"] = df.apply(
        lambda r: get_entity_class(r["instance_of"], r["title"], r["first_paragraph"]),
        axis=1,
    )

    bm25_retriever, bm25_page_ids = load_bm25_index()
    query_tokens = bm25s.tokenize([location_name])
    bm25_docs, bm25_scores = bm25_retriever.retrieve(
        query_tokens,
        k=len(bm25_page_ids),
    )

    # Give every article a score. Articles outside the BM25 index receive 0.
    score_by_page_id = dict.fromkeys(bm25_page_ids, 0.0)
    for doc_id, score in zip(
        np.asarray(bm25_docs).ravel(),
        np.asarray(bm25_scores).ravel(),
    ):
        score_by_page_id[bm25_page_ids[int(doc_id)]] = float(score)
    df["bm25_score"] = df["page_id"].map(score_by_page_id).fillna(0.0)

    # Layer 1: coordinate proximity
    geo = df.dropna(subset=["lat", "lon"]).copy()
    geo["distance_km"] = geo.apply(
        lambda r: haversine_km(user_lat, user_lon, r["lat"], r["lon"]), axis=1
    )
    nearby = geo[geo["distance_km"] <= radius_km].copy()
    nearby["source"] = "coordinates"

    # Layer 2: BM25 fallback for articles without coordinate candidates.
    nearby_ids = set(nearby["page_id"])
    bm25_matches = df[
        (df["bm25_score"] > 0)
        & ~df["page_id"].isin(nearby_ids)
        & df["lat"].isna()
    ].copy()
    bm25_matches["distance_km"] = np.nan
    bm25_matches["source"] = "bm25"

    # Coordinate candidates take precedence; BM25 supplies the missing ones.
    combined = pd.concat([nearby, bm25_matches]).drop_duplicates(
        subset="page_id", keep="first"
    )

    # Composite score
    def composite_score(row):
        s = 0.0
        s += row.get("bm25_score", 0)
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
        "by_century": dict(by_century),
        "by_class": dict(by_class),
        "surprising": surprising,
        "stats": {
            "total_found": len(combined),
            "from_coords": (combined["source"] == "coordinates").sum(),
            "from_bm25": (combined["source"] == "bm25").sum(),
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
        f"{stats['from_bm25']} by BM25 fallback)"
    )
    print(f"{stats['with_dates']} have dates for timeline\n")

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
    index = pd.read_parquet(INDEX_PATH)
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
