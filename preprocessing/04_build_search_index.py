"""
04_build_search_index.py

Joins articles + Wikidata metadata + category info into a single
local_history_index.parquet ready for the local history demo.

One row per article. Columns:
  page_id, title, first_paragraph, full_text
  lat, lon                          — from Wikidata (null if no coordinates)
  birth_date, death_date            — for people
  inception, dissolved              — for places/orgs
  start_time, end_time, point_in_time — for events
  instance_of                       — list of entity types
  country, location                 — from Wikidata
  hop                               — minimum hop from History root (1-4)
  is_list_article, has_stub_category
  canonical_date                    — best available date for timeline
  entity_class                      — simplified type: person/event/place/work/other
"""

import re
import ast
import os
import pandas as pd
import numpy as np
from pathlib import Path

def resolve_data_dir() -> Path:
    """Return the data directory, allowing local or deployed configuration."""
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent.parent.parent / "data",
        Path(__file__).resolve().parent / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()

print("Loading files ...")

articles = pd.read_parquet(
    DATA_DIR / "articles_tagged.parquet",
    columns=[
        "page_id",
        "title",
        "first_paragraph",
        "full_text",
        "is_redirect",
        "is_list_article",
    ],
)
print(f"  Articles:    {len(articles):,}")

wikidata = pd.read_parquet(
    DATA_DIR / "wikidata_metadata.parquet",
    columns=[
        "page_id",
        "lat",
        "lon",
        "birth_date",
        "death_date",
        "inception",
        "dissolved",
        "start_time",
        "end_time",
        "point_in_time",
        "instance_of",
        "occupation",
        "country",
        "location",
        "wikidata_id",
    ],
)
print(f"  Wikidata:    {len(wikidata):,}")

cats = pd.read_parquet(DATA_DIR / "article_categories.parquet", columns=["page_id", "hop"])
# Keep minimum hop per article — closest to History root
min_hop = cats.groupby("page_id")["hop"].min().reset_index()
min_hop.columns = ["page_id", "hop"]
print(f"  Categories:  {len(min_hop):,} unique articles with hop")

# ── Join ──────────────────────────────────────────────────────────────────────

print("\nJoining ...")
df = articles.merge(wikidata, on="page_id", how="left")
df = df.merge(min_hop, on="page_id", how="left")
print(f"  Combined:    {len(df):,} articles")

# ── Canonical date ────────────────────────────────────────────────────────────
# Pick the best single date for timeline display, in priority order


def canonical_date(row):
    """Return the most meaningful date for timeline display."""
    for col in [
        "point_in_time",
        "start_time",
        "birth_date",
        "inception",
        "end_time",
        "death_date",
        "dissolved",
    ]:
        val = row.get(col)
        if pd.notna(val) and val:
            return val
    return None


df["canonical_date"] = df.apply(canonical_date, axis=1)


# Extract year from date strings like "1066-10-20" or "-0044-03-15"
def extract_year(date_str):
    if not date_str or pd.isna(date_str):
        return None
    try:
        # Handle BCE dates (negative years)
        s = str(date_str).strip()
        if s.startswith("-"):
            year_part = s[1:].split("-")[0]
            return -int(year_part)
        else:
            return int(s.split("-")[0])
    except Exception:
        return None


df["year"] = df["canonical_date"].apply(extract_year)

# ── Entity class ──────────────────────────────────────────────────────────────
# Simplify instance_of list into one of 5 classes

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
}


def parse_instance_of(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    try:
        import numpy as np

        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass
    if isinstance(val, str):
        import ast

        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    return []


def entity_class(instance_of):
    types = parse_instance_of(instance_of)
    if not types:
        return "other"
    types_lower = {t.lower() for t in types}
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
    return "other"


df["entity_class"] = df["instance_of"].apply(entity_class)

print(f"\nEntity class breakdown:")
print(df["entity_class"].value_counts().to_string())

# ── Coverage report ───────────────────────────────────────────────────────────

print(f"\nCoverage:")
for col in ["lat", "year", "entity_class", "hop", "country"]:
    if col == "entity_class":
        n = (df[col] != "other").sum()
    elif col == "hop":
        n = df[col].notna().sum()
    else:
        n = df[col].notna().sum()
    print(f"  {col:<20} {n:>7,}  ({n/len(df)*100:.1f}%)")

geo = df["lat"].notna().sum()
print(f"\n  {geo:,} articles have coordinates ({geo/len(df)*100:.1f}%)")
print(f"  {df['year'].notna().sum():,} articles have a year for timeline")

# ── Save ──────────────────────────────────────────────────────────────────────

out = DATA_DIR / "local_history_index.parquet"
df.to_parquet(out, index=False, compression="snappy")
print(f"\nSaved → {out}")
print(f"Columns: {df.columns.tolist()}")
