# localhistory
Turns Wikipedia and Wikidata into an explorable graph of historical events and places.

## Project layout

- `search.py` contains the local-history search and ranking logic.
- `preprocessing/` contains the scripts that build the search index from Wikipedia
  and Wikidata data.
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
```

Script 04 writes `local_history_index.parquet`. Script 05 streams that Parquet
file into `local_history_search.sqlite` and builds a persistent SQLite FTS5
phrase index. Search uses SQLite when the database exists and is newer than the
Parquet source, otherwise it falls back to the slower DuckDB literal scan.

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
