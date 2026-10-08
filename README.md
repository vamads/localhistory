# localhistory
Turn Wikipedia and Wikidata into explorable data of historical places, people, and events.

## Project layout

- `src/localhistory/` contains the API, search, ranking, and runtime helpers.
- `preprocessing/` contains the scripts that build the search index from Wikipedia
  and Wikidata data.
- `scripts/` contains profiling, diagnostic, and data fetching scripts.
- `data/` is local-only and is excluded from Git. It contains raw dumps,
  checkpoints, and generated Parquet files.

## Setup

```bash
pip install -e ".[api,dev]"
```

`uv.lock` pins the complete dependency graph for the project. For reproducible environments, install with `uv` instead of resolving
the dependencies again:

```bash
uv sync --frozen --extra api
```

For the preprocessing environment:

```bash
uv sync --frozen --extra preprocessing
```

Regenerate the lock after changing `pyproject.toml` with `uv lock` and commit
the resulting `uv.lock`.

The inference/API install is intentionally CPU-only: it uses NumPy for the
small candidate-vector similarity calculation and memory-maps the stored
embeddings. Recognized city queries use the precomputed city vectors. This makes returning different query searches much faster. 

For preprocessing, including embedding generation and the Wikipedia/Spark
pipeline, install the separate heavier dependencies (e.g., transformers):

```bash
pip install -e ".[preprocessing]"
```

If you need the fallback that generates an embedding for a query that is not
recognized as a city alias, add the optional model dependencies:

```bash
pip install -e ".[semantic]"
```

For an API-only environment, install the smaller environment:

```bash
pip install -e ".[api]"
```

By default, scripts look for `data/` inside this repository. To use a different
local data directory, set:

```bash
export LOCAL_HISTORY_DATA_DIR=/path/to/localhistory/data
```

## Runtime data

The backend/inference needs these five files in
`LOCAL_HISTORY_DATA_DIR`:

```text
local_history_runtime.sqlite
kalm_embeddings.npy
kalm_embedding_page_ids.npy
city_queries.sqlite
city_query_embeddings.npy
```

`local_history_runtime.sqlite` is the consolidated read-only database created
by preprocessing step 11. It contains the article tables, SQLite FTS search index, citation
index, and editorial-importance scores. The two `kalm_*.npy` files are the
page-id-sorted article embedding matrix and its aligned page IDs. The two city
files provide city aliases, coordinates, and precomputed query vectors, so the
CPU-only runtime does not need KaLM or Torch.

## Preprocessing pipeline

Download the dated Wikimedia inputs first. This includes the bulk Wikipedia
article-content dump (`pages-articles.xml.bz2`), not only the SQL metadata and
link dumps. URLs, sizes, and checksums are recorded in `source_manifest.json`:

```bash
scripts/download_wikimedia_dumps.sh 20250901
export WIKIMEDIA_VERSION=20250901
```

Use `--version latest` only when a moving input is acceptable. The download
shell script uses resumable `curl` or `wget` downloads and then invokes the
Python verifier. The Python script can also be run directly; both expand the
compressed article XML because the PySpark job consumes the uncompressed file.

Run these scripts in order:

```bash
python preprocessing/01_build_category_filter.py
python preprocessing/02_extract_wikipedia.py
python preprocessing/03_fetch_wikidata_metadata.py
python preprocessing/04_build_search_index.py
python preprocessing/05_build_sqlite_search.py
python preprocessing/06_build_kalm_embeddings.py
python preprocessing/07_build_embedding_memmap.py
python preprocessing/08_build_city_query_embeddings.py
python preprocessing/09_build_citation_index.py
python preprocessing/10_build_link_counts.py
python preprocessing/11_build_runtime_database.py
```

### Rebuilding from source data

For a complete rebuild the pipeline needs data from wikidata.E.g., 

```text
enwiki-{version}-page.sql.gz
enwiki-{version}-linktarget.sql.gz
enwiki-{version}-categorylinks.sql.gz
enwiki-{version}-pagelinks.sql.gz
enwiki-{version}-pages-articles.xml
```

For exact reproduction, record the download dates or dump
versions, the Wikidata query date, the Python lock file, the KaLM model
revision, and the generated `runtime_manifest.json`. The scripts currently
refer to `enwiki-latest` and live Wikidata data, so rerunning them later is not necessarily a byte-for-byte reconstruction of the same Wikipedia corpus.

The main PySpark workload is step 2, `preprocessing/02_extract_wikipedia.py`.
It parses the full Wikipedia XML dump in a local Spark session, joins against
`article_to_category.parquet`, and writes `articles.parquet` plus
`article_categories.parquet`. The remaining preprocessing steps are ordinary
Python/NumPy/Pandas/SQLite jobs, except for the embedding model in steps 6 and
8.

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
Script 06 generates resumable KaLM first-paragraph embedding checkpoints.
Script 07 converts those checkpoints into a sorted NumPy memory map.
Search then reads only vectors belonging to the FTS candidates rather than
loading the complete embedding matrix into RAM and accelerator memory.
Script 08 builds a local GeoNames city/alias resolver and precomputes one KaLM
query vector per city. Recognized city searches then avoid both the Nominatim
request and loading KaLM at runtime. The city data is provided by GeoNames
under CC BY 4.0: https://www.geonames.org/
Script 09 extracts only citation-like sentences into a separate SQLite FTS5
index. Search can then look up citation-context matches by city phrase without
scanning or splitting candidate article text at runtime.
Script 10 is a one-time link-data job. It counts links from all normal
Wikipedia articles to normal Wikipedia articles, then writes only Local
History targets to `article_link_counts.parquet`. It also writes metadata to
`article_link_counts.json`; neither output is required until link importance
is added to the runtime ranking. Script 11 imports those scores into the
production database.

When `article_link_counts.parquet` is present, search loads its incoming-link
counts once, applies `log1p`, caps them at the 99th percentile, and adds a small
editorial-prominence bonus to already-retrieved candidates. Link data does not
create candidates or override local text, quality, and geographic relevance.

For deployment, script 11 creates `local_history_runtime.sqlite`, which
contains the article search tables, article FTS index, citation FTS index, and
article importance scores. The API automatically prefers this finalized
database when it exists. The raw search database, citation database, link-count
Parquet file, and large preprocessing artifacts do not need to be deployed.
The embedding `.npy` files remain separate because they are memory-mapped at
query time. `runtime_manifest.json` records the production snapshot metadata.

## Profile search performance

Run the profiler with fixed coordinates to measure cold-start and warm-search
latency without including geocoding network time:

```bash
python scripts/profile_search.py
python scripts/profile_search.py --query Detroit --lat 42.3314 --lon -83.0458 --repeats 5
```

The report includes cProfile hotspots, Python allocation peaks, process RSS,
PyTorch device memory, and cache hit/miss counts. Use `--profile-output
search.prof` to save a cProfile file for later inspection.

## Profile the web API

Measure server launch, the first HTTP search, and repeated warm searches using
the same endpoint as the frontend:

```bash
python scripts/profile_api.py
python scripts/profile_api.py --query Detroit --repeats 5
```

Run this from the `localhistory` repository root. The API also returns a
`Server-Timing` header that separates geocoding, search, serialization, and
total request time; it is visible on `/api/search` in browser developer tools.
