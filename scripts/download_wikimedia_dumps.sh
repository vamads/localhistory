#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for large Wikimedia downloads. The Python script remains
# the canonical verifier and writes source_manifest.json after downloading.

VERSION="${1:-${WIKIMEDIA_VERSION:-latest}}"
DATA_DIR="${LOCAL_HISTORY_DATA_DIR:-data}"
BASE_URL="https://dumps.wikimedia.org/enwiki/${VERSION}"

if command -v curl >/dev/null 2>&1; then
    fetch() {
        curl --fail --location --retry 3 --continue-at - --output "$2" "$1"
    }
elif command -v wget >/dev/null 2>&1; then
    fetch() {
        wget --continue --tries=3 --output-document="$2" "$1"
    }
else
    echo "error: curl or wget is required" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"

for filename in \
    "enwiki-${VERSION}-page.sql.gz" \
    "enwiki-${VERSION}-linktarget.sql.gz" \
    "enwiki-${VERSION}-categorylinks.sql.gz" \
    "enwiki-${VERSION}-pagelinks.sql.gz" \
    "enwiki-${VERSION}-pages-articles.xml.bz2"; do
    destination="${DATA_DIR}/${filename}"
    if [[ -f "$destination" ]]; then
        echo "Already present: ${destination}"
    else
        echo "Downloading: ${filename}"
        fetch "${BASE_URL}/${filename}" "$destination"
    fi
done

compressed_xml="${DATA_DIR}/enwiki-${VERSION}-pages-articles.xml.bz2"
xml="${DATA_DIR}/enwiki-${VERSION}-pages-articles.xml"
if [[ ! -f "$xml" ]]; then
    echo "Decompressing: ${compressed_xml}"
    bzip2 -dk "$compressed_xml"
fi

python scripts/download_wikimedia_dumps.py \
    --version "$VERSION" \
    --data-dir "$DATA_DIR"
