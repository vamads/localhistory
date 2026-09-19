"""Build a page-id-sorted, memory-mapped KaLM embedding matrix.

The source embedding Parquet files are useful checkpoints, but reading and
concatenating every file at search startup allocates the complete embedding
corpus. This script converts them into two aligned NumPy files:

* ``kalm_embedding_page_ids.npy`` contains sorted Wikipedia page IDs.
* ``kalm_embeddings.npy`` contains the corresponding float32 vectors.

Search can binary-search the page IDs and copy only candidate vectors from the
memory map. Run from the localhistory directory after generating embeddings:

    python preprocessing/06_build_embedding_memmap.py
    python preprocessing/06_build_embedding_memmap.py --overwrite
"""

import argparse
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def resolve_data_dir() -> Path:
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data",
        Path(__file__).resolve().parent.parent / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()
SOURCE_DIR = DATA_DIR / "kalm_first_paragraph_embeddings"
MATRIX_PATH = DATA_DIR / "kalm_embeddings.npy"
PAGE_IDS_PATH = DATA_DIR / "kalm_embedding_page_ids.npy"
FILENAME_PATTERN = re.compile(r"embeddings_(\d+)_(\d+)\.parquet$")


def embedding_file_key(path: Path) -> tuple[int, int]:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected embedding filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def parquet_embedding_files(source_dir: Path) -> list[Path]:
    return sorted(source_dir.glob("embeddings_*.parquet"), key=embedding_file_key)


def read_embedding_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path, columns=["page_id", "embedding"])
    page_ids = frame["page_id"].to_numpy(dtype=np.int64, copy=True)
    if frame.empty:
        return page_ids, np.empty((0, 0), dtype=np.float32)
    matrix = np.vstack(frame["embedding"].to_numpy()).astype(np.float32, copy=False)
    return page_ids, matrix


def build_memmap(
    source_dir: Path,
    matrix_path: Path,
    page_ids_path: Path,
    *,
    reorder_batch_size: int,
    overwrite: bool,
) -> None:
    files = parquet_embedding_files(source_dir)
    if not files:
        raise FileNotFoundError(f"No embedding Parquet files found in {source_dir}")

    existing_outputs = [path for path in (matrix_path, page_ids_path) if path.exists()]
    if existing_outputs and not overwrite:
        names = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Output already exists: {names}\nPass --overwrite to rebuild it."
        )

    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    first_ids, first_matrix = read_embedding_file(files[0])
    if first_matrix.ndim != 2 or first_matrix.shape[1] == 0:
        raise ValueError(f"Could not determine embedding dimension from {files[0]}")
    dimension = first_matrix.shape[1]

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_building = matrix_path.with_name(matrix_path.name + ".building")
    ids_building = page_ids_path.with_name(page_ids_path.name + ".building")
    unsorted_matrix_path = matrix_path.with_name(matrix_path.name + ".unsorted.building")
    temporary_paths = (matrix_building, ids_building, unsorted_matrix_path)
    for path in temporary_paths:
        if path.exists():
            path.unlink()

    print(f"Source: {source_dir}")
    print(f"Files: {len(files):,}")
    print(f"Embeddings: {total_rows:,} x {dimension:,} float32")
    print(f"Output matrix: {matrix_path}")

    started = time.perf_counter()
    try:
        raw_matrix = np.lib.format.open_memmap(
            unsorted_matrix_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, dimension),
        )
        raw_ids = np.empty(total_rows, dtype=np.int64)

        offset = 0
        for file_number, path in enumerate(files):
            if file_number == 0:
                page_ids, matrix = first_ids, first_matrix
            else:
                page_ids, matrix = read_embedding_file(path)
            if matrix.ndim != 2 or matrix.shape[1] != dimension:
                raise ValueError(
                    f"Expected {dimension} dimensions in {path}, found {matrix.shape}"
                )
            end = offset + len(page_ids)
            raw_ids[offset:end] = page_ids
            raw_matrix[offset:end] = matrix
            offset = end
            print(f"  staged {offset:,}/{total_rows:,}")

        if offset != total_rows:
            raise RuntimeError(f"Expected {total_rows:,} rows, staged {offset:,}")
        raw_matrix.flush()

        order = np.argsort(raw_ids, kind="stable")
        sorted_ids = raw_ids[order]
        if len(sorted_ids) > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
            duplicate = int(sorted_ids[1:][sorted_ids[1:] == sorted_ids[:-1]][0])
            raise ValueError(f"Duplicate embedding page_id: {duplicate}")

        final_matrix = np.lib.format.open_memmap(
            matrix_building,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, dimension),
        )
        final_ids = np.lib.format.open_memmap(
            ids_building,
            mode="w+",
            dtype=np.int64,
            shape=(total_rows,),
        )
        final_ids[:] = sorted_ids
        final_ids.flush()

        for start in range(0, total_rows, reorder_batch_size):
            end = min(start + reorder_batch_size, total_rows)
            final_matrix[start:end] = raw_matrix[order[start:end]]
            print(f"  sorted {end:,}/{total_rows:,}")
        final_matrix.flush()

        del final_matrix
        del final_ids
        del raw_matrix
        unsorted_matrix_path.unlink()

        os.replace(matrix_building, matrix_path)
        os.replace(ids_building, page_ids_path)
    except BaseException:
        for path in temporary_paths:
            if path.exists():
                path.unlink()
        raise

    elapsed = time.perf_counter() - started
    size_gib = (matrix_path.stat().st_size + page_ids_path.stat().st_size) / (1024 ** 3)
    print(f"Complete in {elapsed:.1f} seconds")
    print(f"Combined size: {size_gib:.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--matrix-output", type=Path, default=MATRIX_PATH)
    parser.add_argument("--page-ids-output", type=Path, default=PAGE_IDS_PATH)
    parser.add_argument("--reorder-batch-size", type=int, default=5_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.reorder_batch_size <= 0:
        parser.error("--reorder-batch-size must be positive")

    build_memmap(
        args.source_dir.expanduser().resolve(),
        args.matrix_output.expanduser().resolve(),
        args.page_ids_output.expanduser().resolve(),
        reorder_batch_size=args.reorder_batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
