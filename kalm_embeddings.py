import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path("/Users/vicenteamado/Documents/LLM/history_ML/data")
OUT_DIR = BASE / "kalm_first_paragraph_embeddings"
OUT_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 64  # articles per encode() call
SAVE_EVERY = 5_000  # save a parquet file every N articles
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    trust_remote_code=True,
    device=DEVICE,
)
model.max_seq_length = 4096  # a few mis-parsed articles have 40k+ char "first_paragraph" text (12k+ tokens) that OOMs the attention mask otherwise; mask memory scales with seq_len^2 so this caps it around ~4GB at batch_size=64

# ── Load articles ─────────────────────────────────────────────────────────────
print("Loading articles...")
df = pd.read_parquet(
    BASE / "articles.parquet", columns=["page_id", "first_paragraph"]
).dropna(subset=["first_paragraph"])
df = df[df["first_paragraph"].str.len() > 50].reset_index(drop=True)
print(f"  {len(df):,} articles to embed")

# ── Check for existing checkpoints ───────────────────────────────────────────
existing = list(OUT_DIR.glob("embeddings_*.parquet"))
if existing:
    last_idx = max(int(p.stem.split("_")[-1]) for p in existing)
    df = df[df.index > last_idx].reset_index(drop=True)
    print(f"  Resuming from article {last_idx:,} — {len(df):,} remaining")
else:
    last_idx = -1
    print("  Starting fresh")

# ── Embed in chunks ───────────────────────────────────────────────────────────
buffer_ids = []
buffer_embs = []
total_done = 0
t0 = time.time()

for i in range(0, len(df), SAVE_EVERY):
    chunk = df.iloc[i : i + SAVE_EVERY]
    texts = chunk["first_paragraph"].tolist()
    page_ids = chunk["page_id"].tolist()

    # Embed the chunk
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        device=DEVICE,
    )

    # Save this chunk immediately
    global_start = last_idx + 1 + i
    global_end = global_start + len(chunk) - 1
    out_path = OUT_DIR / f"embeddings_{global_start}_{global_end}.parquet"

    chunk_df = pd.DataFrame(
        {
            "page_id": page_ids,
            "embedding": list(embs.astype(np.float32)),
        }
    )
    chunk_df.to_parquet(out_path, index=False)

    if DEVICE == "mps":
        torch.mps.empty_cache()

    total_done += len(chunk)
    elapsed = time.time() - t0
    rate = total_done / elapsed
    remaining = (len(df) - total_done) / rate / 3600

    print(
        f"  Saved {out_path.name} | "
        f"{total_done:,}/{len(df):,} done | "
        f"{rate:.0f} articles/sec | "
        f"~{remaining:.1f}h remaining"
    )

print(f"\nDone. {total_done:,} articles embedded.")
print(f"Files saved to {OUT_DIR}")
