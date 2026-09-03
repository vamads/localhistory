"""
02_extract_wikipedia.py

Production PySpark job that processes the full English Wikipedia XML dump
and outputs two clean Parquet files for the HistoryGraph project.

What this script does in plain English:
  1. Start a Spark session using all cores on your Mac
  2. Load your filter file (article_to_category.parquet) — the list of
     article page_ids that belong to your 42k history categories
  3. Stream the 108GB Wikipedia XML dump, parsing each article
  4. Extract: page_id, title, first paragraph, full clean text
  5. Keep only articles that are in your history filter file
  6. Write two Parquet files:
       articles.parquet           — one row per article with text
       article_categories.parquet — one row per article-category edge

Runtime: ~2-4 hours on M1 Pro with 10 cores
Outputs: ~15-30GB total Parquet

Usage:
    python 02_extract_wikipedia.py

Make sure to close Chrome and other heavy apps before running.
"""

import bz2
import os
import re
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col
from pyspark.sql.types import LongType, StringType, BooleanType, StructType, StructField

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# Change these paths if your files are in a different location.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def resolve_data_dir() -> Path:
    """Return the data directory, allowing local or deployed configuration."""
    configured = os.getenv("LOCAL_HISTORY_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent.parent.parent / "data",
        Path(__file__).resolve().parent / "data",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


DATA_DIR = resolve_data_dir()

XML_PATH = DATA_DIR / "enwiki-latest-pages-articles.xml"  # 108GB decompressed
FILTER_PATH = DATA_DIR / "article_to_category.parquet"  # built by bfs_from_dumps.py

OUT_ARTICLES = DATA_DIR / "articles.parquet"
OUT_ARTICLE_CATS = DATA_DIR / "article_categories.parquet"

# How many cores to use. -1 means "use all available cores" (recommended).
# On your M1 Pro with 10 cores this will use all 10.
NUM_CORES = -1

# How much RAM to give Spark. 10GB as you decided.
SPARK_MEMORY = "10g"

# How many partitions to split the data into for parallel processing.
# Rule of thumb: 2-4x your core count. With 10 cores, 40 is good.
NUM_PARTITIONS = 40


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: START SPARK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_spark_session() -> SparkSession:
    """
    Start a local Spark session using all CPU cores.

    local[*] means: run on this machine, use all available cores.
    On Databricks later, you remove this line entirely — Databricks
    creates the session for you automatically.
    """
    cores = "*" if NUM_CORES == -1 else str(NUM_CORES)

    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName("HistoryGraph-Wikipedia-Pipeline")
        .config("spark.driver.memory", SPARK_MEMORY)
        # maxResultSize: max data that can be collected back to the driver.
        # Prevents out-of-memory errors when doing large joins.
        .config("spark.driver.maxResultSize", "4g")
        # shuffle.partitions: how many partitions to use after a join/groupBy.
        # Default is 200, which is too high for local mode.
        .config("spark.sql.shuffle.partitions", str(NUM_PARTITIONS))
        # Enable arrow-based columnar data transfer between JVM and Python.
        # Makes pandas↔Spark conversions much faster.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark started.")
    print(f"  Cores:   {spark.sparkContext.defaultParallelism}")
    print(f"  Memory:  {SPARK_MEMORY}")
    print(f"  UI:      http://localhost:4040")
    return spark


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: LOAD THE FILTER FILE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_filter(spark: SparkSession):
    """
    Load article_to_category.parquet — the file bfs_from_dumps.py created.
    This tells us which article page_ids belong to history categories.

    We load it as a Python set for fast O(1) lookup inside the XML parser.
    We also keep the full DataFrame for the article_categories output.

    Why a set and not a Spark join here?
    The XML parsing happens inside mapPartitions (Python workers).
    Those workers can't access Spark DataFrames — they only see plain Python.
    So we broadcast the filter as a Python set to all workers.
    """
    print(f"\nLoading filter file: {FILTER_PATH.name} ...", flush=True)

    df = spark.read.parquet(str(FILTER_PATH))

    # Collect just the page_ids into a Python set.
    # This set gets sent to every worker so they can filter during parsing.
    history_page_ids = set(
        row.page_id for row in df.select("page_id").distinct().collect()
    )

    print(f"  {len(history_page_ids):,} unique history article page_ids loaded")
    print(f"  Hop breakdown:")
    df.groupBy("hop").count().orderBy("hop").show()

    return history_page_ids, df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: XML PARSING FUNCTIONS
# These run inside Spark workers — imports must be inside the functions.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def extract_text(wikitext: str) -> tuple[str, str]:
    """
    Parse raw wikitext and return (first_paragraph, full_text).

    mwparserfromhell handles the heavy lifting:
    - Strips [[wikilinks]], {{templates}}, '''bold''', ''italic''
    - Removes category tags, reference tags, infoboxes
    - Returns clean plain text

    first_paragraph: the first meaningful paragraph (>50 chars).
                     This is Wikipedia's own summary of the article.
                     Perfect for embeddings.

    full_text: all paragraphs joined, with section headers removed.
               Useful for RAG later.
    """
    import mwparserfromhell

    if not wikitext:
        return "", ""

    try:
        wikicode = mwparserfromhell.parse(wikitext)
        plain = wikicode.strip_code(normalize=True, collapse=True)

        # Split into paragraphs, skip empty ones and very short ones
        # (short ones are usually template artifacts like "See also")
        paragraphs = [p.strip() for p in plain.split("\n") if len(p.strip()) > 50]

        first_paragraph = paragraphs[0] if paragraphs else ""
        full_text = "\n\n".join(paragraphs)

        return first_paragraph, full_text

    except Exception:
        # mwparserfromhell can fail on malformed wikitext — just skip
        return "", ""


def parse_xml_chunk(args):
    """
    Parse a chunk of the Wikipedia XML file and yield article rows.

    This function runs on each Spark worker in parallel.
    Each worker gets a (start_byte, end_byte) range of the XML file
    and parses just that portion.

    Why mapPartitions instead of a UDF?
    UDFs work row-by-row. mapPartitions works on a batch of rows at once,
    which lets us open the file once per partition instead of once per row.
    Much more efficient for file I/O.

    Yields tuples of:
      (page_id, title, first_paragraph, full_text, is_redirect, redirect_target)
    """
    import mwxml
    import mwparserfromhell

    xml_path, history_ids, start_byte, end_byte = args

    # Open the XML file and seek to our chunk's start position
    with open(xml_path, "rb") as f:
        # Read just our chunk
        f.seek(start_byte)
        chunk = f.read(end_byte - start_byte)

    # mwxml needs a file-like object — wrap the chunk in a StringIO
    # We need to add XML wrapper tags so it's valid XML
    import io

    xml_wrapper = (
        b'<mediawiki xmlns="http://www.mediawiki.org/xml/mediawiki">'
        + chunk
        + b"</mediawiki>"
    )

    try:
        dump = mwxml.Dump.from_file(io.BytesIO(xml_wrapper))
    except Exception:
        return  # malformed chunk — skip

    for page in dump:
        # Only process articles (namespace 0)
        # Namespace 14 = categories, 1 = talk pages, etc.
        if page.namespace != 0:
            continue

        page_id = page.id
        title = page.title or ""

        # Handle redirects — log them but don't extract text
        if page.redirect:
            # Only yield if this redirect is in history categories
            if page_id in history_ids:
                yield (page_id, title, "", "", True, page.redirect)
            continue

        # Filter early — skip articles not in history categories
        # This is the most important optimization: we avoid parsing
        # wikitext for the ~6M articles we don't care about
        if page_id not in history_ids:
            continue

        # Get the latest revision's wikitext
        wikitext = ""
        for revision in page:
            wikitext = revision.text or ""
            break  # only need the latest revision

        first_para, full_text = extract_text(wikitext)

        # Skip articles with no extractable text
        if not first_para:
            continue

        yield (page_id, title, first_para, full_text, False, None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: SPLIT THE XML FILE INTO CHUNKS FOR PARALLEL PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_page_boundaries(xml_path: Path, num_chunks: int) -> list[tuple[int, int]]:
    """
    Split the XML file into chunks at <page> boundaries.

    Why can't Spark just split the XML file automatically?
    XML files aren't splittable — if you cut one in the middle of a <page>
    tag, neither half is valid XML. We need to find clean split points
    at </page> boundaries so each chunk is self-contained.

    Returns a list of (start_byte, end_byte) pairs — one per chunk.
    Each chunk contains complete <page>...</page> blocks.
    """
    print(f"\nFinding page boundaries in {xml_path.name} ...", flush=True)

    file_size = xml_path.stat().st_size
    chunk_size = file_size // num_chunks
    boundaries = [0]

    with open(xml_path, "rb") as f:
        for i in range(1, num_chunks):
            # Seek to the approximate chunk boundary
            target = i * chunk_size
            f.seek(target)

            # Read forward until we find a </page> tag
            # This ensures we don't cut in the middle of an article
            buffer = b""
            while True:
                data = f.read(4096)
                if not data:
                    break
                buffer += data
                idx = buffer.find(b"</page>")
                if idx != -1:
                    # Split after the </page> tag
                    boundaries.append(target + idx + len(b"</page>"))
                    break

    boundaries.append(file_size)

    # Remove duplicates and sort
    boundaries = sorted(set(boundaries))
    chunks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    print(f"  Split into {len(chunks)} chunks")
    print(f"  Avg chunk size: {file_size // len(chunks) / 1e9:.2f} GB")
    return chunks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: RUN THE PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_pipeline(
    spark: SparkSession,
    history_page_ids: set,
    filter_df,
) -> None:
    """
    Main pipeline: parse XML → filter → write Parquet.
    """
    sc = spark.sparkContext

    # -- Split XML into chunks for parallel processing --
    chunks = get_page_boundaries(XML_PATH, num_chunks=NUM_PARTITIONS)

    # Build the args list — one tuple per chunk
    # Each tuple contains everything the worker needs
    xml_path_str = str(XML_PATH)
    args = [(xml_path_str, history_page_ids, start, end) for start, end in chunks]

    print(f"\nParsing XML with {len(chunks)} parallel chunks ...", flush=True)
    print(
        f"  This will take 2-4 hours. Progress shows in Spark UI: http://localhost:4040"
    )
    print(f"  You can monitor progress there while it runs.\n", flush=True)

    # -- Define output schema --
    # Spark needs to know the column types upfront when converting from RDD
    ARTICLE_SCHEMA = StructType(
        [
            StructField("page_id", LongType(), nullable=False),
            StructField("title", StringType(), nullable=False),
            StructField("first_paragraph", StringType(), nullable=True),
            StructField("full_text", StringType(), nullable=True),
            StructField("is_redirect", BooleanType(), nullable=True),
            StructField("redirect_target", StringType(), nullable=True),
        ]
    )

    # -- Create RDD from args and run mapPartitions --
    # sc.parallelize() distributes the args list across workers
    # flatMap(parse_xml_chunk) calls parse_xml_chunk on each element
    # and flattens the yielded rows into one big RDD
    args_rdd = sc.parallelize(args, numSlices=len(chunks))
    articles_rdd = args_rdd.flatMap(parse_xml_chunk)

    # -- Convert RDD to DataFrame --
    articles_df = spark.createDataFrame(articles_rdd, schema=ARTICLE_SCHEMA)

    # Cache since we'll write two different outputs from this DataFrame
    # Without cache, Spark would re-parse the XML twice
    articles_df.cache()

    # -- Trigger execution and report --
    print("Counting articles (triggers full execution) ...", flush=True)
    total = articles_df.count()
    print(f"\n  Total history articles extracted: {total:,}")

    # Show a sample so you can verify the output looks right
    print("\nSample articles:")
    articles_df.select("page_id", "title", "first_paragraph").show(10, truncate=80)

    # -- Write articles.parquet --
    print(f"\nWriting {OUT_ARTICLES.name} ...", flush=True)
    articles_df.write.mode("overwrite").parquet(str(OUT_ARTICLES))
    print(f"  Saved → {OUT_ARTICLES}")

    # -- Write article_categories.parquet --
    # Join articles with the filter_df to get category + hop for each article
    # This gives us the edge list: article → category
    print(f"\nWriting {OUT_ARTICLE_CATS.name} ...", flush=True)

    article_cats_df = (
        articles_df.select("page_id", "title")
        .join(
            broadcast(filter_df),  # broadcast the small side of the join
            on="page_id",
            how="inner",
        )
        .select("page_id", "title", "category", "hop")
    )

    article_cats_df.write.mode("overwrite").parquet(str(OUT_ARTICLE_CATS))

    n_edges = article_cats_df.count()
    print(f"  Saved → {OUT_ARTICLE_CATS}")
    print(f"  {n_edges:,} article→category edges")

    # -- Final summary --
    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  articles.parquet:            {total:,} articles")
    print(f"  article_categories.parquet:  {n_edges:,} edges")
    print(f"\nNext steps:")
    print(f"  1. Validate output in a notebook")
    print(f"  2. Upload to Databricks DBFS")
    print(f"  3. Embed articles with KaLM mini v2.5")
    print(f"{'='*60}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    print("=" * 60)
    print("HistoryGraph Wikipedia Pipeline")
    print("=" * 60)
    print(f"  XML:    {XML_PATH}")
    print(f"  Filter: {FILTER_PATH}")
    print(f"  Output: {OUT_ARTICLES}")
    print(f"          {OUT_ARTICLE_CATS}")

    # Verify input files exist before starting Spark
    for path in [XML_PATH, FILTER_PATH]:
        if not path.exists():
            print(f"\n[ERROR] File not found: {path}")
            raise SystemExit(1)

    spark = create_spark_session()

    try:
        history_page_ids, filter_df = load_filter(spark)
        run_pipeline(spark, history_page_ids, filter_df)
    finally:
        spark.stop()
        print("\nSparkSession stopped.")


if __name__ == "__main__":
    main()
