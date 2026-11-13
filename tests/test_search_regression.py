"""Opt-in integration checks for important city/article search expectations.

Run with the local data and model artifacts enabled::

    LOCAL_HISTORY_RUN_SEARCH_REGRESSION=1 \
      uv run --extra api --extra semantic pytest -q tests/test_search_regression.py

The expected-title lists are intentionally easy to edit as search quality
requirements evolve.
"""

from __future__ import annotations

import os
import re
import time

import pytest

from localhistory.city_queries import resolve_city_query
from localhistory.search import DATA_DIR, search_local_history

# Add or remove titles here as the desired search contract becomes clearer.
SEARCH_EXPECTATIONS = {
    "Paris, France": [
        "Napoleon Bonaparte",
        "Louis XVI",
        "St. Bartholomew's Day massacre",
        "Notre-Dame de Paris",
        "Clovis I",
    ],
    "Ann Arbor, Michigan": [
        "University of Michigan",
        "Huron River",
        "Anishinaabe",
        "Treaty of Detroit",
        "John Allen (pioneer)",
    ],
    "Detroit, Michigan": [
        "Ford Motor Company",
        "Antoine de la Mothe Cadillac",
        "Detroit bankruptcy",
        "Motown",
        "Detroit River",
        "Great Fire of 1805",
    ],
    "A Coruña, Spain": [
        "Ferdinand II of León",
        "Tower of Hercules",
        "Brigantes",
        "Galicia (Spain)",
        "Battle of Corunna",
        "María Pita",
    ],
}


def normalize_title(value: str) -> str:
    value = value.replace("_", " ").casefold()
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip()


def regression_data_available() -> bool:
    required = (
        DATA_DIR / "city_queries.sqlite",
        DATA_DIR / "local_history_runtime.sqlite",
        DATA_DIR / "kalm_embedding_page_ids.npy",
        DATA_DIR / "kalm_embeddings.npy",
    )
    return all(path.exists() for path in required)


@pytest.mark.search_regression
@pytest.mark.skipif(
    os.getenv("LOCAL_HISTORY_RUN_SEARCH_REGRESSION") != "1",
    reason="opt-in integration test; set LOCAL_HISTORY_RUN_SEARCH_REGRESSION=1",
)
def test_key_city_searches_return_expected_articles():
    if not regression_data_available():
        pytest.skip(f"local search data is unavailable under {DATA_DIR}")

    rerank = os.getenv("LOCAL_HISTORY_REGRESSION_RERANK", "1") != "0"
    failures = []
    for query, expected_titles in SEARCH_EXPECTATIONS.items():
        started = time.perf_counter()
        print(f"\n[{query}] starting search...", flush=True)
        city = resolve_city_query(query)
        assert city is not None, f"City is not recognized: {query}"

        output = search_local_history(
            location_name=city.canonical_query,
            user_lat=city.latitude,
            user_lon=city.longitude,
            radius_km=50.0,
            top_n=150,
            bm25_min_score=1.0,
            dense_top_k=200,
            population=city.population,
            country_name=city.country_name,
            geoname_id=city.geoname_id,
            rerank=rerank,
        )
        titles = [
            normalize_title(str(title)) for title in output["results"]["title"].tolist()
        ]
        title_set = set(titles)
        missing = [
            title
            for title in expected_titles
            if normalize_title(title) not in title_set
        ]
        if missing:
            failures.append(
                f"{query}: missing {missing}; top results were "
                f"{output['results']['title'].head(20).tolist()}"
            )
        print(
            f"[{query}] finished in {time.perf_counter() - started:.2f}s; "
            f"scope={output['stats'].get('candidate_scope')}; "
            f"graph_candidates={output['stats'].get('city_graph_candidates')}; "
            f"missing={missing}",
            flush=True,
        )

    assert not failures, "\n".join(failures)
