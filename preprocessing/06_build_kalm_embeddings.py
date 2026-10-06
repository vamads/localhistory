"""Generate resumable KaLM embeddings for article first paragraphs.

Run from the localhistory directory after ``02_extract_wikipedia.py`` has
written ``articles.parquet``::

    python preprocessing/06_build_kalm_embeddings.py
    python preprocessing/06_build_kalm_embeddings.py --overwrite

The output Parquet files are checkpoints consumed by
``07_build_embedding_memmap.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"


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
ARTICLES_PATH = DATA_DIR / "articles.parquet"
OUTPUT_DIR = DATA_DIR / "kalm_first_paragraph_embeddings"


def select_device(requested: str | None) -> str:
    if requested:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_embeddings(
    articles_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
    save_every: int,
    device: str | None,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("embeddings_*.parquet"))
    if existing and overwrite:
        for path in existing:
            path.unlink()
        existing = []

    articles = pd.read_parquet(
        articles_path, columns=["page_id", "first_paragraph"]
    ).dropna(subset=["first_paragraph"])
    articles = articles[articles["first_paragraph"].str.len() > 50].reset_index(drop=True)
    if existing:
        last_index = max(int(path.stem.split("_")[-1]) for path in existing)
        articles = articles.iloc[last_index + 1 :].reset_index(drop=True)
        offset = last_index + 1
        print(f"Resuming after article {last_index:,}: {len(articles):,} remaining")
    else:
        offset = 0
        print(f"Starting fresh: {len(articles):,} articles to embed")

    if articles.empty:
        print("No embeddings to generate.")
        return

    selected_device = select_device(device)
    print(f"Loading {MODEL_NAME} on {selected_device}...")
    model = SentenceTransformer(
        MODEL_NAME, trust_remote_code=True, device=selected_device
    )
    model.max_seq_length = 4096

    for start in range(0, len(articles), save_every):
        chunk = articles.iloc[start : start + save_every]
        embeddings = model.encode(
            chunk["first_paragraph"].tolist(),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=selected_device,
        )
        global_start = offset + start
        global_end = global_start + len(chunk) - 1
        output_path = output_dir / f"embeddings_{global_start}_{global_end}.parquet"
        pd.DataFrame(
            {
                "page_id": chunk["page_id"].tolist(),
                "embedding": list(np.asarray(embeddings, dtype=np.float32)),
            }
        ).to_parquet(output_path, index=False)
        if selected_device == "mps":
            torch.mps.empty_cache()
        print(f"Saved {output_path.name} ({global_end + 1:,}/{offset + len(articles):,})")

    print(f"Complete. Checkpoints saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=ARTICLES_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.save_every <= 0:
        parser.error("--batch-size and --save-every must be positive")

    build_embeddings(
        args.articles.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        batch_size=args.batch_size,
        save_every=args.save_every,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
