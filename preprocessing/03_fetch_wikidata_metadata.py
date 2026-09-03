"""
03_fetch_wikidata_metadata.py

Fetches structured metadata from Wikidata for your 440k history articles.

Properties fetched:
  Pass 1 — Spatial + temporal:
    P625  coordinates
    P569  date of birth
    P570  date of death
    P571  inception (when something was founded/created)
    P576  dissolved/abolished date
    P580  start time
    P582  end time
    P585  point in time (for battles, elections, etc.)

  Pass 2 — Entity type + context:
    P31   instance of (what kind of thing is this)
    P106  occupation (for people)
    P17   country
    P27   country of citizenship
    P131  located in administrative territory

Checkpointing: saves progress every 50 batches so you can resume if interrupted.

Usage:
    python 03_fetch_wikidata_metadata.py

Output:
    wikidata_metadata.parquet
"""

import time
import json
import os
import requests
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

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
ARTICLES_PATH = DATA_DIR / "articles.parquet"
CHECKPOINT = DATA_DIR / "wikidata_checkpoint.json"  # saves progress
OUT_PATH = DATA_DIR / "wikidata_metadata.parquet"

BATCH_SIZE = 100  # keep batches small to avoid URI length limits
SLEEP_BETWEEN = 2.0  # seconds between requests
CHECKPOINT_EVERY = 50  # save to disk every N batches
MAX_RETRIES = 3

SPARQL_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "HistoryGraph/1.0 (academic research; contact: history-graph-project)",
    "Accept": "application/sparql-results+json",
}

# ── Load articles ─────────────────────────────────────────────────────────────

print("Loading articles ...")
df = pd.read_parquet(ARTICLES_PATH, columns=["page_id", "title"])
titles = df["title"].tolist()
page_ids = df["page_id"].tolist()
title_to_id = dict(zip(df["title"], df["page_id"]))
print(f"  {len(titles):,} articles")

# ── Resume from checkpoint ───────────────────────────────────────────────────

results = {}  # title -> dict of properties

if CHECKPOINT.exists():
    print(f"  Resuming from checkpoint: {CHECKPOINT}")
    with open(CHECKPOINT) as f:
        checkpoint_data = json.load(f)
    results = checkpoint_data.get("results", {})
    start_batch = checkpoint_data.get("last_batch", 0)
    print(
        f"  {len(results):,} articles already fetched, resuming from batch {start_batch}"
    )
else:
    start_batch = 0
    print(f"  Starting fresh")

# ── SPARQL query ──────────────────────────────────────────────────────────────


def build_query(batch_titles: list[str]) -> str:
    """
    Query for Pass 1 + Pass 2 properties in one request per batch.
    Uses OPTIONAL so missing properties return null rather than dropping the row.
    """
    # Escape quotes in titles
    escaped = [t.replace('"', '\\"').replace("\\", "\\\\") for t in batch_titles]
    titles_str = "\n    ".join(f'"{t}"@en' for t in escaped)

    return f"""
SELECT ?title ?item
       ?lat ?lon
       ?birthDate ?deathDate
       ?inception ?dissolved
       ?startTime ?endTime ?pointInTime
       ?instanceOfLabel
       ?occupationLabel
       ?countryLabel
       ?citizenshipLabel
       ?locationLabel
WHERE {{
  VALUES ?title {{ {titles_str} }}

  # Link Wikipedia article title to Wikidata item
  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?title .

  # Pass 1: spatial + temporal (all optional)
  OPTIONAL {{
    ?item wdt:P625 ?coords .
    BIND(geof:latitude(?coords)  AS ?lat)
    BIND(geof:longitude(?coords) AS ?lon)
  }}
  OPTIONAL {{ ?item wdt:P569 ?birthDate . }}
  OPTIONAL {{ ?item wdt:P570 ?deathDate . }}
  OPTIONAL {{ ?item wdt:P571 ?inception . }}
  OPTIONAL {{ ?item wdt:P576 ?dissolved . }}
  OPTIONAL {{ ?item wdt:P580 ?startTime . }}
  OPTIONAL {{ ?item wdt:P582 ?endTime . }}
  OPTIONAL {{ ?item wdt:P585 ?pointInTime . }}

  # Pass 2: entity type + context (labels auto-resolved)
  OPTIONAL {{ ?item wdt:P31  ?instanceOf . }}
  OPTIONAL {{ ?item wdt:P106 ?occupation . }}
  OPTIONAL {{ ?item wdt:P17  ?country . }}
  OPTIONAL {{ ?item wdt:P27  ?citizenship . }}
  OPTIONAL {{ ?item wdt:P131 ?location . }}

  # Auto-resolve labels for entity-type properties
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
    ?instanceOf  rdfs:label ?instanceOfLabel .
    ?occupation  rdfs:label ?occupationLabel .
    ?country     rdfs:label ?countryLabel .
    ?citizenship rdfs:label ?citizenshipLabel .
    ?location    rdfs:label ?locationLabel .
  }}
}}
"""


# ── SPARQL runner ─────────────────────────────────────────────────────────────


def run_sparql(query: str) -> list[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            # Use POST to avoid 414 URI Too Long errors
            resp = requests.post(
                SPARQL_URL,
                data={"query": query, "format": "json"},
                headers={
                    **HEADERS,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=90,
            )
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"\n    Rate limited — waiting {wait}s ...")
                time.sleep(wait)
                continue
            if resp.status_code == 500:
                print(f"\n    Server error 500 — retrying in 10s ...")
                time.sleep(10)
                continue
            resp.raise_for_status()
            return resp.json()["results"]["bindings"]
        except requests.exceptions.Timeout:
            print(f"\n    Timeout on attempt {attempt+1} — retrying ...")
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"\n    Error attempt {attempt+1}: {e}")
            time.sleep(5 * (attempt + 1))
    print("    Max retries exceeded — skipping batch")
    return []


# ── Parse a single binding row ────────────────────────────────────────────────


def get_val(binding: dict, key: str) -> str | None:
    return binding.get(key, {}).get("value")


def parse_date(val: str | None) -> str | None:
    """Strip Wikidata datetime format to just the date part."""
    if not val:
        return None
    # Wikidata dates look like: +1815-06-18T00:00:00Z
    return val.lstrip("+").split("T")[0]


def parse_binding(b: dict) -> tuple[str, dict]:
    title = get_val(b, "title")
    return title, {
        "wikidata_id": get_val(b, "item"),
        "lat": float(get_val(b, "lat")) if get_val(b, "lat") else None,
        "lon": float(get_val(b, "lon")) if get_val(b, "lon") else None,
        "birth_date": parse_date(get_val(b, "birthDate")),
        "death_date": parse_date(get_val(b, "deathDate")),
        "inception": parse_date(get_val(b, "inception")),
        "dissolved": parse_date(get_val(b, "dissolved")),
        "start_time": parse_date(get_val(b, "startTime")),
        "end_time": parse_date(get_val(b, "endTime")),
        "point_in_time": parse_date(get_val(b, "pointInTime")),
        "instance_of": get_val(
            b, "instanceOfLabel"
        ),  # will be collected as list in merge
        "occupation": get_val(b, "occupationLabel"),
        "country": get_val(b, "countryLabel"),
        "citizenship": get_val(b, "citizenshipLabel"),
        "location": get_val(b, "locationLabel"),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

n_batches = (len(titles) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\nQuerying Wikidata: {n_batches} batches of {BATCH_SIZE}")
print(
    f"Estimated time: {n_batches * (SLEEP_BETWEEN + 3) / 3600:.1f}-{n_batches * (SLEEP_BETWEEN + 8) / 3600:.1f} hours"
)
print(f"Checkpointing every {CHECKPOINT_EVERY} batches\n")

for batch_num in range(start_batch, n_batches):
    i = batch_num * BATCH_SIZE
    batch_titles = titles[i : i + BATCH_SIZE]

    query = build_query(batch_titles)
    bindings = run_sparql(query)

    # Parse and merge — one article can have multiple rows
    # (e.g. multiple occupations) so we merge by taking first non-null value
    # Exception: instance_of collects ALL values as a list
    for b in bindings:
        title, props = parse_binding(b)
        if not title:
            continue
        if title not in results:
            results[title] = props
            # Start instance_of as a list
            if props.get("instance_of"):
                results[title]["instance_of"] = [props["instance_of"]]
            else:
                results[title]["instance_of"] = []
        else:
            # Merge: fill in any None values from new row
            for k, v in props.items():
                if k == "instance_of":
                    # Collect all instance_of values as a list
                    if v and v not in results[title]["instance_of"]:
                        results[title]["instance_of"].append(v)
                elif results[title].get(k) is None and v is not None:
                    results[title][k] = v

    # Progress
    if (batch_num + 1) % 10 == 0 or batch_num == 0:
        pct = (batch_num + 1) / n_batches * 100
        print(
            f"  Batch {batch_num+1:>4}/{n_batches} ({pct:4.1f}%) | "
            f"{len(results):>7,} articles with data",
            flush=True,
        )

    # Checkpoint
    if (batch_num + 1) % CHECKPOINT_EVERY == 0:
        with open(CHECKPOINT, "w") as f:
            json.dump({"results": results, "last_batch": batch_num + 1}, f)
        print(f"  ✓ Checkpoint saved at batch {batch_num+1}")

    time.sleep(SLEEP_BETWEEN)

# ── Save final output ─────────────────────────────────────────────────────────

print(f"\nDone. {len(results):,} articles with Wikidata metadata.")

rows = []
for title, props in results.items():
    page_id = title_to_id.get(title)
    if page_id:
        rows.append({"page_id": int(page_id), "title": title, **props})

out_df = pd.DataFrame(rows)

# Coverage report
print(f"\nCoverage report:")
for col in [
    "lat",
    "birth_date",
    "death_date",
    "inception",
    "point_in_time",
    "instance_of",
    "country",
]:
    if col in out_df.columns:
        n = out_df[col].notna().sum()
        print(f"  {col:<20} {n:>7,}  ({n/len(out_df)*100:.1f}%)")

out_df.to_parquet(OUT_PATH, index=False, compression="snappy")
print(f"\nSaved → {OUT_PATH}")

# Clean up checkpoint
if CHECKPOINT.exists():
    CHECKPOINT.unlink()
    print("Checkpoint file removed.")
