"""Profile local-history search speed and memory usage.

Examples:
    python profile_search.py
    python profile_search.py --query Detroit --lat 42.3314 --lon -83.0458 --repeats 5
    python profile_search.py --query "Ann Arbor" --profile-output search.prof

The first measured search includes memory-map initialization. Later searches
measure the warm path with those resources cached in the process.
Geocoding is intentionally excluded; coordinates are supplied on the command
line so the results reflect search performance rather than network latency.
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import resource
import time
import tracemalloc
from pathlib import Path

import pandas as pd

from search import (
    SQLITE_SEARCH_PATH,
    kalm_device,
    load_candidate_index,
    load_citation_counts,
    load_kalm_embeddings,
    load_kalm_model,
    search_local_history,
)


def rss_bytes() -> int:
    """Return the process high-water RSS in bytes on macOS and Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def sync_device() -> None:
    """Wait for asynchronous GPU work before recording timings."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except (ImportError, AttributeError, RuntimeError):
        pass


def torch_memory() -> dict[str, int | str] | None:
    try:
        import torch

        device = kalm_device()
        result: dict[str, int | str] = {"device": device}
        if device.startswith("cuda"):
            result.update(
                allocated_bytes=int(torch.cuda.memory_allocated()),
                reserved_bytes=int(torch.cuda.memory_reserved()),
            )
        elif device.startswith("mps"):
            result["allocated_bytes"] = int(torch.mps.current_allocated_memory())
        return result
    except (ImportError, AttributeError, RuntimeError, StopIteration):
        return None


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def run_search(args: argparse.Namespace, index: pd.DataFrame) -> dict:
    sync_device()
    started = time.perf_counter()
    output = search_local_history(
        location_name=args.query,
        user_lat=args.lat,
        user_lon=args.lon,
        radius_km=args.radius,
        top_n=args.top_n,
        bm25_min_score=1.0,
        kalm_top_k=200,
        index=index,
    )
    sync_device()
    return {
        "seconds": time.perf_counter() - started,
        "results": len(output["results"]),
        "candidates": output["stats"]["total_found"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Ann Arbor, Michigan")
    parser.add_argument("--lat", type=float, default=42.2808)
    parser.add_argument("--lon", type=float, default=-83.7430)
    parser.add_argument("--radius", type=float, default=50.0, help="Search radius in km")
    parser.add_argument("--top-n", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-output", type=Path, default=None)
    args = parser.parse_args()

    print(f"Index: {SQLITE_SEARCH_PATH}")
    print(f"Query: {args.query!r} ({args.lat}, {args.lon}), radius={args.radius} km")
    print("Loading matching article candidates...")
    index = load_candidate_index(args.query, args.lat, args.lon, args.radius)
    print(f"Candidate articles: {len(index):,}")
    print(f"Initial RSS: {format_bytes(rss_bytes())}")

    tracemalloc.start()
    cold_profiler = cProfile.Profile()
    cold_profiler.enable()
    cold = run_search(args, index)
    cold_profiler.disable()
    current, peak = tracemalloc.get_traced_memory()
    print("\nCold search")
    print(f"  elapsed: {cold['seconds']:.3f} s")
    print(f"  results/candidates: {cold['results']}/{cold['candidates']}")
    print(f"  Python allocations: current={format_bytes(current)}, peak={format_bytes(peak)}")
    print(f"  Process RSS high-water: {format_bytes(rss_bytes())}")
    print(f"  PyTorch memory: {json.dumps(torch_memory())}")

    if args.profile_output:
        cold_profiler.dump_stats(args.profile_output)
        print(f"  cProfile data: {args.profile_output}")
    print("\nCold-search hotspots (cumulative time)")
    stats_stream = io.StringIO()
    pstats.Stats(cold_profiler, stream=stats_stream).sort_stats("cumulative").print_stats(25)
    print(stats_stream.getvalue())

    warm_times = []
    for _ in range(max(0, args.repeats)):
        result = run_search(args, index)
        warm_times.append(result["seconds"])
    if warm_times:
        average = sum(warm_times) / len(warm_times)
        print("Warm searches")
        print("  times: " + ", ".join(f"{value:.3f} s" for value in warm_times))
        print(f"  average: {average:.3f} s")
        print(f"  final RSS high-water: {format_bytes(rss_bytes())}")
        print(f"  final PyTorch memory: {json.dumps(torch_memory())}")

    print("\nCache sizes")
    for name, function in (
        ("Candidate index", load_candidate_index),
        ("Citation index", load_citation_counts),
        ("KaLM model", load_kalm_model),
        ("KaLM embeddings", load_kalm_embeddings),
    ):
        info = function.cache_info()
        print(f"  {name}: hits={info.hits}, misses={info.misses}, size={info.currsize}/{info.maxsize}")
    tracemalloc.stop()


if __name__ == "__main__":
    main()
