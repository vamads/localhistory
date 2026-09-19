"""Build a GeoNames city resolver and precomputed KaLM query embeddings.

The default GeoNames tier contains cities with populations above 15,000 plus
capitals. One canonical query is embedded per city; alternate names are kept
as SQLite aliases pointing at that vector.

Run from the localhistory directory:

    python preprocessing/07_build_city_query_embeddings.py
    python preprocessing/07_build_city_query_embeddings.py --overwrite
"""

import argparse
from collections.abc import Iterable
import os
from pathlib import Path
import sqlite3
import sys
import time

import geonamescache
import numpy as np
import torch


PACKAGE_PARENT = Path(__file__).resolve().parent.parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from localhistory.city_queries import (  # noqa: E402
    CITY_QUERY_DATABASE_PATH,
    CITY_QUERY_EMBEDDINGS_PATH,
    normalize_city_alias,
)


KALM_MODEL_NAME = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
SUPPORTED_POPULATION_TIERS = (500, 1_000, 5_000, 15_000)

SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE cities (
    geoname_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    canonical_query TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    country_code TEXT NOT NULL,
    country_name TEXT NOT NULL,
    admin1_code TEXT NOT NULL,
    admin1_name TEXT NOT NULL,
    population INTEGER NOT NULL,
    embedding_row INTEGER NOT NULL UNIQUE
);

CREATE TABLE aliases (
    normalized_alias TEXT NOT NULL,
    geoname_id INTEGER NOT NULL REFERENCES cities(geoname_id),
    PRIMARY KEY (normalized_alias, geoname_id)
) WITHOUT ROWID;

CREATE INDEX aliases_by_city ON aliases(geoname_id);
"""


def embedding_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def canonical_city_query(
    city: dict,
    country_name: str,
    state_names: dict[str, str],
) -> tuple[str, str]:
    state_name = state_names.get(city["admin1code"], "")
    qualifier = state_name if city["countrycode"] == "US" and state_name else country_name
    return f"{city['name']}, {qualifier}", state_name


def city_aliases(
    city: dict,
    *,
    canonical_query: str,
    country_name: str,
    state_name: str,
) -> set[str]:
    """Generate lookup aliases without embedding each spelling separately."""
    aliases = {
        normalize_city_alias(city["name"]),
        normalize_city_alias(canonical_query),
        normalize_city_alias(f"{city['name']}, {country_name}"),
        normalize_city_alias(f"{city['name']}, {city['countrycode']}"),
    }
    if city["countrycode"] == "US" and state_name:
        aliases.update(
            {
                normalize_city_alias(f"{city['name']}, {state_name}"),
                normalize_city_alias(f"{city['name']}, {city['admin1code']}"),
                normalize_city_alias(
                    f"{city['name']}, {state_name}, {country_name}"
                ),
                normalize_city_alias(
                    f"{city['name']}, {city['admin1code']}, US"
                ),
            }
        )

    for alternate_name in city.get("alternatenames", []):
        if 2 <= len(alternate_name) <= 100:
            aliases.add(normalize_city_alias(alternate_name))
    return {alias for alias in aliases if len(alias) >= 2}


def flush_aliases(
    connection: sqlite3.Connection,
    rows: list[tuple[str, int]],
) -> None:
    if not rows:
        return
    connection.executemany(
        "INSERT OR IGNORE INTO aliases(normalized_alias, geoname_id) VALUES (?, ?)",
        rows,
    )
    rows.clear()


def build_city_database(
    path: Path,
    cities: list[dict],
    countries: dict[str, dict],
    states: dict[str, dict],
    *,
    minimum_population: int,
) -> list[str]:
    state_names = {code: state["name"] for code, state in states.items()}
    canonical_queries: list[str] = []
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("model_name", KALM_MODEL_NAME),
                ("minimum_population", str(minimum_population)),
                ("city_count", str(len(cities))),
            ],
        )

        alias_rows: list[tuple[str, int]] = []
        for embedding_row, city in enumerate(cities):
            country = countries.get(city["countrycode"], {})
            country_name = country.get("name", city["countrycode"])
            canonical_query, state_name = canonical_city_query(
                city,
                country_name,
                state_names,
            )
            canonical_queries.append(canonical_query)
            connection.execute(
                """
                INSERT INTO cities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(city["geonameid"]),
                    city["name"],
                    canonical_query,
                    float(city["latitude"]),
                    float(city["longitude"]),
                    city["countrycode"],
                    country_name,
                    city["admin1code"],
                    state_name,
                    int(city["population"]),
                    embedding_row,
                ),
            )
            alias_rows.extend(
                (alias, int(city["geonameid"]))
                for alias in city_aliases(
                    city,
                    canonical_query=canonical_query,
                    country_name=country_name,
                    state_name=state_name,
                )
            )
            if len(alias_rows) >= 50_000:
                flush_aliases(connection, alias_rows)

        flush_aliases(connection, alias_rows)
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        alias_count = connection.execute("SELECT count(*) FROM aliases").fetchone()[0]
        print(f"Aliases: {alias_count:,}")
    finally:
        connection.close()
    return canonical_queries


def embedding_batches(values: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(values), batch_size):
        yield start, values[start : start + batch_size]


def build(
    database_path: Path,
    embeddings_path: Path,
    *,
    minimum_population: int,
    batch_size: int,
    device: str | None,
    overwrite: bool,
    limit: int | None,
) -> None:
    existing = [path for path in (database_path, embeddings_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}\nPass --overwrite to rebuild it.")

    cache = geonamescache.GeonamesCache(min_city_population=minimum_population)
    countries = cache.get_countries()
    states = cache.get_us_states()
    cities = sorted(
        cache.get_cities().values(),
        key=lambda city: int(city["geonameid"]),
    )
    if limit is not None:
        cities = sorted(
            cities,
            key=lambda city: (-int(city["population"]), int(city["geonameid"])),
        )[:limit]
        cities.sort(key=lambda city: int(city["geonameid"]))

    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_building = database_path.with_name(database_path.name + ".building")
    embeddings_building = embeddings_path.with_name(embeddings_path.name + ".building")
    temporary_paths = (database_building, embeddings_building)
    for path in temporary_paths:
        if path.exists():
            path.unlink()

    print(f"Cities: {len(cities):,}")
    print(f"Population tier: {minimum_population:,}")
    started = time.perf_counter()
    try:
        canonical_queries = build_city_database(
            database_building,
            cities,
            countries,
            states,
            minimum_population=minimum_population,
        )

        from sentence_transformers import SentenceTransformer

        selected_device = embedding_device(device)
        print(f"Loading {KALM_MODEL_NAME} on {selected_device}...")
        model = SentenceTransformer(
            KALM_MODEL_NAME,
            trust_remote_code=True,
            device=selected_device,
        )
        model.max_seq_length = 4096

        first = model.encode(
            canonical_queries[:1],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        dimension = first.shape[1]
        embeddings = np.lib.format.open_memmap(
            embeddings_building,
            mode="w+",
            dtype=np.float32,
            shape=(len(canonical_queries), dimension),
        )
        embeddings[0] = first[0]

        for start, batch in embedding_batches(canonical_queries[1:], batch_size):
            output_start = start + 1
            output_end = output_start + len(batch)
            embeddings[output_start:output_end] = model.encode(
                batch,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            ).astype(np.float32)
            print(f"  embedded {output_end:,}/{len(canonical_queries):,}")
        embeddings.flush()
        del embeddings

        os.replace(database_building, database_path)
        os.replace(embeddings_building, embeddings_path)
    except BaseException:
        for path in temporary_paths:
            if path.exists():
                path.unlink()
        raise

    elapsed = time.perf_counter() - started
    total_mib = (database_path.stat().st_size + embeddings_path.stat().st_size) / (1024 ** 2)
    print(f"Complete in {elapsed:.1f} seconds")
    print(f"Combined size: {total_mib:.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-output", type=Path, default=CITY_QUERY_DATABASE_PATH)
    parser.add_argument("--embeddings-output", type=Path, default=CITY_QUERY_EMBEDDINGS_PATH)
    parser.add_argument(
        "--minimum-population",
        type=int,
        choices=SUPPORTED_POPULATION_TIERS,
        default=15_000,
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--limit", type=int, help="Build only the largest N cities for testing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    build(
        args.database_output.expanduser().resolve(),
        args.embeddings_output.expanduser().resolve(),
        minimum_population=args.minimum_population,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
