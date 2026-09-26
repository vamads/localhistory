# localhistory
Turns Wikipedia and Wikidata into an explorable graph of historical events and places.

## Project layout

- `search.py` contains the local-history search and ranking logic.
- `preprocessing/` contains the scripts that build the search index from Wikipedia
  and Wikidata data.
- `notebooks/explore_wikipedia_links.ipynb` inspects the Wikipedia page/link dumps
  and prototypes incoming-link importance features.
- `data/` is local-only and is excluded from Git. It contains raw dumps,
  checkpoints, and generated Parquet files.

## Setup

```bash
pip install -r requirements.txt
```

By default, scripts look for `data/` inside this repository. To use a different
local data directory, set:

```bash
export LOCAL_HISTORY_DATA_DIR=/path/to/localhistory/data
```

## Preprocessing pipeline

Run these scripts in order:

```bash
python preprocessing/01_build_category_filter.py
python preprocessing/02_extract_wikipedia.py
python preprocessing/03_fetch_wikidata_metadata.py
python preprocessing/04_build_search_index.py
python preprocessing/05_build_sqlite_search.py
python preprocessing/06_build_embedding_memmap.py
python preprocessing/07_build_city_query_embeddings.py
python preprocessing/08_build_citation_index.py
python preprocessing/09_build_link_counts.py
```

Script 04 writes `local_history_index.parquet`. Script 05 streams that Parquet
file into `local_history_search.sqlite` and builds a persistent SQLite FTS5
phrase index. FTS5 supplies both candidate retrieval and weighted BM25 ranking
across title, first paragraph, and full text. The SQLite indexes are required
at runtime; search reports the appropriate rebuild command if one is missing
or stale. Article details and map coordinates are also read from this database,
so the web API does not load the Parquet article index at runtime.
Candidate retrieval unions two independent groups: every article inside the
coordinate radius, and articles containing an exact qualified-place phrase
such as `"Jackson Michigan"` or `"Jackson MI"`. Bare city-word and full-text
proximity matches are intentionally excluded to avoid namesake false positives.
Script 06 converts the embedding checkpoints into a sorted NumPy memory map.
Search then reads only vectors belonging to the FTS candidates rather than
loading the complete embedding matrix into RAM and accelerator memory.
Script 07 builds a local GeoNames city/alias resolver and precomputes one KaLM
query vector per city. Recognized city searches then avoid both the Nominatim
request and loading KaLM at runtime. The city data is provided by GeoNames
under CC BY 4.0: https://www.geonames.org/
Script 08 extracts only citation-like sentences into a separate SQLite FTS5
index. Search can then look up citation-context matches by city phrase without
scanning or splitting candidate article text at runtime.
Script 09 is a one-time link-data job. It counts links from all normal
Wikipedia articles to normal Wikipedia articles, then writes only Local
History targets to `article_link_counts.parquet`. It also writes metadata to
`article_link_counts.json`; neither output is required until link importance
is added to the runtime ranking.

When `article_link_counts.parquet` is present, search loads its incoming-link
counts once, applies `log1p`, caps them at the 99th percentile, and adds a small
editorial-prominence bonus to already-retrieved candidates. Link data does not
create candidates or override local text, quality, and geographic relevance.

## Profile search performance

Run the profiler with fixed coordinates to measure cold-start and warm-search
latency without including geocoding network time:

```bash
python profile_search.py
python profile_search.py --query Detroit --lat 42.3314 --lon -83.0458 --repeats 5
```

The report includes cProfile hotspots, Python allocation peaks, process RSS,
PyTorch device memory, and cache hit/miss counts. Use `--profile-output
search.prof` to save a cProfile file for later inspection.

## Profile the web API

Measure server launch, the first HTTP search, and repeated warm searches using
the same endpoint as the frontend:

```bash
python localhistory/profile_api.py
python localhistory/profile_api.py --query Detroit --repeats 5
```

Run this from the `history_ML` repository root. The API also returns a
`Server-Timing` header that separates geocoding, search, serialization, and
total request time; it is visible on `/api/search` in browser developer tools.
