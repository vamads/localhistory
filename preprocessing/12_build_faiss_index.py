"""Build a persisted FAISS index for normalized article embeddings.

Run after preprocessing/07_build_embedding_memmap.py:

    python preprocessing/12_build_faiss_index.py
    python preprocessing/12_build_faiss_index.py --overwrite

The article embeddings are normalized, so inner-product search is cosine
similarity. The exact IndexFlatIP index preserves the current NumPy ranking.
"""

import argparse
import os
from pathlib import Path

import numpy as np


def data_dir() -> Path:
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "FAISS is required. Install the API dependencies with: "
            "uv sync --extra api"
        ) from exc

    base = data_dir()
    matrix_path = base / "kalm_embeddings.npy"
    page_ids_path = base / "kalm_embedding_page_ids.npy"
    output_path = base / "kalm_embeddings.faiss"
    building_path = output_path.with_suffix(".faiss.building")

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Index already exists: {output_path}; use --overwrite")
    if not matrix_path.exists() or not page_ids_path.exists():
        raise FileNotFoundError(
            "Missing embedding memmaps. Run preprocessing/07_build_embedding_memmap.py first."
        )

    matrix = np.load(matrix_path, mmap_mode="r")
    page_ids = np.load(page_ids_path, mmap_mode="r")
    if matrix.ndim != 2 or page_ids.ndim != 1 or len(matrix) != len(page_ids):
        raise ValueError(f"Embedding shapes do not match: {matrix.shape}, {page_ids.shape}")

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(matrix.shape[1]))
    for start in range(0, len(matrix), args.batch_size):
        end = min(start + args.batch_size, len(matrix))
        vectors = np.asarray(matrix[start:end], dtype=np.float32, order="C")
        ids = np.asarray(page_ids[start:end], dtype=np.int64, order="C")
        index.add_with_ids(vectors, ids)
        print(f"Indexed {end:,}/{len(matrix):,} embeddings")

    faiss.write_index(index, str(building_path))
    os.replace(building_path, output_path)
    print(f"Wrote {output_path} ({index.ntotal:,} vectors)")


if __name__ == "__main__":
    main()
