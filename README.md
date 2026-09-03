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
```

The final script writes `local_history_index.parquet`, which `search.py` reads.
