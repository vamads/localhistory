"""Download and verify the Wikimedia inputs used by the preprocessing pipeline.

Examples::

    python scripts/download_wikimedia_dumps.py --version 20250901
    python scripts/download_wikimedia_dumps.py --version latest

The selected version is also written to ``source_manifest.json``. Set
``WIKIMEDIA_VERSION`` to the same value when running preprocessing scripts.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


FILES = {
    "page": (".sql.gz", "enwiki-{version}-page.sql.gz"),
    "linktarget": (".sql.gz", "enwiki-{version}-linktarget.sql.gz"),
    "categorylinks": (".sql.gz", "enwiki-{version}-categorylinks.sql.gz"),
    "pagelinks": (".sql.gz", "enwiki-{version}-pagelinks.sql.gz"),
    "pages_articles": (".xml.bz2", "enwiki-{version}-pages-articles.xml.bz2"),
}


def download(url: str, destination: Path, user_agent: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def remote_sha1(url: str, user_agent: str) -> str | None:
    request = Request(url + ".sha1", headers={"User-Agent": user_agent})
    try:
        with urlopen(request) as response:
            first_field = response.read().decode("ascii").split()[0]
    except Exception:
        return None
    return first_field if len(first_field) == 40 else None


def decompress_xml(compressed: Path, output: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with bz2.open(compressed, "rb") as source, output.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=os.getenv("WIKIMEDIA_VERSION", "latest"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--user-agent", default="localhistory-data-fetcher/0.1")
    args = parser.parse_args()

    data_dir = (args.data_dir or Path(os.getenv("LOCAL_HISTORY_DATA_DIR", "data"))).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://dumps.wikimedia.org/enwiki/{args.version}"
    manifest = {
        "source": "Wikimedia English Wikipedia SQL and bulk article dumps",
        "version": args.version,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    for key, (_, filename_template) in FILES.items():
        filename = filename_template.format(version=args.version)
        url = f"{base_url}/{filename}"
        compressed_path = data_dir / filename
        if not compressed_path.exists():
            print(f"Downloading {url}")
            sha256, size = download(url, compressed_path, args.user_agent)
        else:
            sha256, size = hash_file(compressed_path)

        expected_sha1 = remote_sha1(url, args.user_agent)
        if expected_sha1 is not None and sha1_file(compressed_path) != expected_sha1:
            raise RuntimeError(f"Checksum mismatch for {compressed_path}")
        entry = {
            "url": url,
            "path": compressed_path.name,
            "size_bytes": size,
            "sha256": sha256,
            "remote_sha1": expected_sha1,
        }
        if key == "pages_articles":
            xml_path = data_dir / filename.removesuffix(".bz2")
            if not xml_path.exists():
                print(f"Decompressing {compressed_path.name}")
                xml_sha256, xml_size = decompress_xml(compressed_path, xml_path)
            else:
                xml_sha256, xml_size = hash_file(xml_path)
            entry["decompressed"] = {
                "path": xml_path.name,
                "size_bytes": xml_size,
                "sha256": xml_sha256,
            }
        manifest["files"][key] = entry

    (data_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {data_dir / 'source_manifest.json'}")


if __name__ == "__main__":
    main()
