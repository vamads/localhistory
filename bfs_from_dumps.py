"""
bfs_from_dumps.py

Builds a Wikipedia category graph from three SQL dumps and BFS from a seed.
Also saves article→category membership for use in the production pipeline.

Schema (new MediaWiki):
  page(page_id, page_namespace, page_title, ...)
  linktarget(lt_id, lt_namespace, lt_title)
  categorylinks(cl_from, cl_sortkey, cl_timestamp, cl_sortkey_prefix,
                cl_type, cl_collation_id, cl_target_id)

Passes:
  1. page.sql.gz        → {page_id: title}  ns=14  (category name lookup)
  2. linktarget.sql.gz  → {lt_id: title}    ns=14  (parent category lookup)
  3. categorylinks.sql.gz → {parent_name: {child_page_ids}}  subcat + page edges
  4. BFS from SEED up to MAX_HOPS
  5. Save outputs

Outputs (all written to BASE dir):
  categories_by_hop.json          {hop: [category_name, ...]}
  categories_by_hop.txt           human-readable summary
  article_to_category.parquet     (page_id, category, hop) — filter file for pipeline
"""

import gzip
import re
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

def resolve_data_dir() -> Path:
    """Return the data directory, allowing local or deployed configuration."""
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()
PAGE_SQL = DATA_DIR / "enwiki-latest-page.sql.gz"
LT_SQL = DATA_DIR / "enwiki-latest-linktarget.sql.gz"
CL_SQL = DATA_DIR / "enwiki-latest-categorylinks.sql.gz"

SEED     = "History"
MAX_HOPS = 4          # hops 1-4 = 42,148 categories; hop 5+ is noise

OUT_JSON = DATA_DIR / "categories_by_hop.json"
OUT_TXT = DATA_DIR / "categories_by_hop.txt"
OUT_PARQUET = DATA_DIR / "article_to_category.parquet"

# ── Helpers ───────────────────────────────────────────────────────────────────

def unescape(s: str) -> str:
    return s.replace("\\'", "'").replace("''", "'").replace("\\\\", "\\")

def stream_inserts(path: Path, table: str):
    """Yield INSERT lines from a gzipped SQL dump."""
    needle = f"INSERT INTO `{table}`"
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(needle):
                yield line

# ── Pass 1: page table → {page_id: title} for ns=14 ─────────────────────────

PAGE_RE = re.compile(r"\((\d+),(14),'((?:[^'\\]|\\.)*)'")

def load_page_table(path: Path) -> dict[str, str]:
    id_to_title: dict[str, str] = {}
    print(f"Pass 1: {path.name} ...", flush=True)
    for i, line in enumerate(stream_inserts(path, "page")):
        for m in PAGE_RE.finditer(line):
            id_to_title[m.group(1)] = unescape(m.group(3))
        if i % 1_000 == 0:
            print(f"  {i:>6,} INSERT lines | {len(id_to_title):>7,} category pages", end="\r")
    print(f"\n  Done — {len(id_to_title):,} category pages (ns=14)")
    return id_to_title

# ── Pass 2: linktarget → {lt_id: title} for ns=14 ───────────────────────────

LT_RE = re.compile(r"\((\d+),(14),'((?:[^'\\]|\\.)*)'")

def load_linktarget(path: Path) -> dict[str, str]:
    lt_to_title: dict[str, str] = {}
    print(f"Pass 2: {path.name} ...", flush=True)
    for i, line in enumerate(stream_inserts(path, "linktarget")):
        for m in LT_RE.finditer(line):
            lt_to_title[m.group(1)] = unescape(m.group(3))
        if i % 1_000 == 0:
            print(f"  {i:>6,} INSERT lines | {len(lt_to_title):>7,} category targets", end="\r")
    print(f"\n  Done — {len(lt_to_title):,} category link targets (ns=14)")
    return lt_to_title

# ── Pass 3: categorylinks → name graph + article membership ──────────────────

CL_RE = re.compile(
    r"\((\d+),"                     # cl_from      (child page_id)
    r"'[^']*',"                     # cl_sortkey   (binary, skip)
    r"'[^']*',"                     # cl_timestamp
    r"'[^']*',"                     # cl_sortkey_prefix
    r"'(page|subcat|file)',"        # cl_type
    r"\d+,"                         # cl_collation_id
    r"(\d+)\)"                      # cl_target_id (parent lt_id)
)

def load_category_graph(
    path: Path,
    id_to_title: dict[str, str],
    lt_to_title: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Returns:
      parent_to_children     {parent_category_name: {child_category_names}}
      parent_to_article_ids  {parent_category_name: {article_page_id_strings}}
    """
    parent_to_children:    dict[str, set[str]] = defaultdict(set)
    parent_to_article_ids: dict[str, set[str]] = defaultdict(set)

    print(f"Pass 3: {path.name} ...", flush=True)
    for i, line in enumerate(stream_inserts(path, "categorylinks")):
        for m in CL_RE.finditer(line):
            cl_from   = m.group(1)
            cl_type   = m.group(2)
            cl_target = m.group(3)

            parent_name = lt_to_title.get(cl_target)
            if not parent_name:
                continue

            if cl_type == "subcat":
                child_name = id_to_title.get(cl_from)
                if child_name:
                    parent_to_children[parent_name].add(child_name)

            elif cl_type == "page":
                # cl_from is an article page_id (ns=0) — store as string for now
                parent_to_article_ids[parent_name].add(cl_from)

        if i % 1_000 == 0:
            total_edges = sum(len(v) for v in parent_to_children.values())
            print(f"  {i:>6,} INSERT lines | {len(parent_to_children):>7,} cat parents "
                  f"| {total_edges:>9,} subcat edges", end="\r")

    total_subcat = sum(len(v) for v in parent_to_children.values())
    total_arts   = sum(len(v) for v in parent_to_article_ids.values())
    print(f"\n  Done — {len(parent_to_children):,} parent categories, "
          f"{total_subcat:,} subcat edges, "
          f"{total_arts:,} article memberships across "
          f"{len(parent_to_article_ids):,} categories")
    return dict(parent_to_children), dict(parent_to_article_ids)

# ── Pass 4: BFS ───────────────────────────────────────────────────────────────

def bfs(
    graph: dict[str, set[str]],
    seed: str,
    max_hops: int,
) -> dict[int, list[str]]:
    if seed not in graph:
        print(f"\n[ERROR] '{seed}' not found as a parent in the graph.")
        candidates = sorted(k for k in graph if "history" in k.lower())[:30]
        print(f"  Candidates containing 'history': {candidates}")
        raise SystemExit(1)

    print(f"\nPass 4: BFS from '{seed}', max {max_hops} hops ...", flush=True)
    visited  = {seed}
    by_hop   = {0: [seed]}
    frontier = {seed}

    for hop in range(1, max_hops + 1):
        next_frontier = set()
        for parent in frontier:
            for child in graph.get(parent, set()):
                if child not in visited:
                    visited.add(child)
                    next_frontier.add(child)
        by_hop[hop] = sorted(next_frontier)
        print(f"  Hop {hop}: {len(next_frontier):>6,} new categories  "
              f"(cumulative: {len(visited):,})")
        frontier = next_frontier
        if not frontier:
            print("  Frontier empty — graph exhausted.")
            break

    return by_hop

# ── Pass 5: Save outputs ──────────────────────────────────────────────────────

def save_outputs(
    by_hop: dict[int, list[str]],
    parent_to_article_ids: dict[str, set[str]],
) -> None:
    print(f"\nPass 5: Saving outputs ...", flush=True)

    # 5a. categories_by_hop.json
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in by_hop.items()}, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {OUT_JSON}")

    # 5b. categories_by_hop.txt
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for hop in sorted(by_hop.keys()):
            cats = by_hop[hop]
            f.write(f"\n{'='*60}\n")
            f.write(f"HOP {hop}  ({len(cats)} categories)\n")
            f.write(f"{'='*60}\n")
            for cat in cats:
                f.write(f"  {cat}\n")
    print(f"  Saved → {OUT_TXT}")

    # 5c. article_to_category.parquet
    # Build flat (page_id, category, hop) rows for all history categories only
    print(f"  Building article→category membership ...", flush=True)

    # Map every history category to its hop number
    cat_to_hop: dict[str, int] = {}
    for hop, cats in by_hop.items():
        for cat in cats:
            cat_to_hop[cat] = int(hop)

    rows = []
    for category, article_ids in parent_to_article_ids.items():
        hop = cat_to_hop.get(category)
        if hop is None:
            continue  # not a history category — skip
        for page_id in article_ids:
            rows.append({
                "page_id":  int(page_id),
                "category": category,
                "hop":      hop,
            })

    df = pd.DataFrame(rows, columns=["page_id", "category", "hop"])

    # An article can belong to multiple history categories.
    # Keep all memberships (don't deduplicate) so the pipeline can build
    # multiple article→category edges in the graph.
    df = df.sort_values(["page_id", "hop"]).reset_index(drop=True)

    df.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    print(f"  Saved → {OUT_PARQUET}")
    print(f"  {len(df):,} article→category rows")
    print(f"  {df['page_id'].nunique():,} unique article page_ids")
    print(f"\n  Articles per hop:")
    hop_counts = df.groupby("hop")["page_id"].nunique()
    print(hop_counts.to_string())

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    id_to_title           = load_page_table(PAGE_SQL)
    lt_to_title           = load_linktarget(LT_SQL)
    graph, parent_to_arts = load_category_graph(CL_SQL, id_to_title, lt_to_title)
    by_hop                = bfs(graph, SEED, MAX_HOPS)
    save_outputs(by_hop, parent_to_arts)

    print(f"\nAll done.")
    print(f"  {OUT_JSON.name:<40} {sum(len(v) for v in by_hop.values()):,} categories")
    print(f"  {OUT_PARQUET.name:<40} ready for production PySpark pipeline")


if __name__ == "__main__":
    main()
