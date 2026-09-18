"""Measure API launch, cold-search, and warm-search latency.

Run from the history_ML repository root:

    python localhistory/profile_api.py
    python localhistory/profile_api.py --query Detroit --repeats 5
"""

import argparse
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def wait_until_ready(url: str, timeout: float) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except (URLError, TimeoutError):
            time.sleep(0.1)
    raise TimeoutError(f"API did not become ready within {timeout:.0f} seconds")


def timed_request(url: str) -> tuple[float, str]:
    started = time.perf_counter()
    with urlopen(url, timeout=300) as response:
        response.read()
        server_timing = response.headers.get("Server-Timing", "unavailable")
    return time.perf_counter() - started, server_timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Ann Arbor, Michigan")
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    search_url = base_url + "/api/search?" + urlencode(
        {"q": args.query, "radius_km": args.radius}
    )
    launched = time.perf_counter()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "localhistory.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
    )

    try:
        ready_seconds = wait_until_ready(
            base_url + "/api/health", args.startup_timeout
        )
        cold_seconds, cold_server_timing = timed_request(search_url)
        launch_to_result = time.perf_counter() - launched

        warm_results = [timed_request(search_url) for _ in range(args.repeats)]
        warm_seconds = [result[0] for result in warm_results]

        print(f"API ready: {ready_seconds:.3f} s")
        print(f"Cold HTTP search: {cold_seconds:.3f} s")
        print(f"  Server-Timing: {cold_server_timing}")
        print(f"Launch to first result: {launch_to_result:.3f} s")
        if warm_seconds:
            print("Warm HTTP searches: " + ", ".join(f"{value:.3f} s" for value in warm_seconds))
            print(f"Warm average: {sum(warm_seconds) / len(warm_seconds):.3f} s")
            print(f"  Last Server-Timing: {warm_results[-1][1]}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
