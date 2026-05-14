"""
ingestion/ingest.py

Full ingestion pipeline — run once to populate ChromaDB.
Idempotent: already-ingested volumes are skipped via ingested_files.json.

Usage:
    python -m ingestion.ingest
  or:
    venv\\Scripts\\python.exe -m ingestion.ingest

Prerequisites:
    - .env file with GOOGLE_API_KEY set
    - data/{book}/vol{n}.pdf files present
    - ChromaDB and google-generativeai installed (pip install -r requirements.txt)
"""

import os
import json
import sys
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

import chromadb

from ingestion.loader import load_pdf_text
from ingestion.chunker import chunk_text
from ingestion.embedder import HadithEmbedder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "kutub_al_sittah")
INGESTED_FILES_PATH = "ingestion/ingested_files.json"

# Exact volume counts that are physically on disk.
# If you add more volumes later, update the number here.
BOOKS = {
    "bukhari":   {"display": "Sahih al-Bukhari",  "volumes": 9},
    "muslim":    {"display": "Sahih Muslim",        "volumes": 7},
    "abu_dawud": {"display": "Sunan Abu Dawud",     "volumes": 5},
    "tirmidhi":  {"display": "Jami at-Tirmidhi",   "volumes": 6},
    "nasai":     {"display": "Sunan an-Nasai",      "volumes": 5},
    "ibn_majah": {"display": "Sunan Ibn Majah",     "volumes": 5},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ingested_log() -> dict:
    """Load the idempotency tracker. Returns {} if it doesn't exist yet."""
    if os.path.exists(INGESTED_FILES_PATH):
        with open(INGESTED_FILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_ingested_log(log: dict) -> None:
    """Persist the idempotency tracker to disk."""
    os.makedirs(os.path.dirname(INGESTED_FILES_PATH), exist_ok=True)
    with open(INGESTED_FILES_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def _deduplicate_ids(ids: list[str]) -> list[str]:
    """
    If the same hadith number was parsed twice in a volume (formatting edge
    case), append _dup1, _dup2 etc. to make every ID unique.
    Logs a warning for any duplicates found.
    """
    counts = Counter(ids)
    duplicates = {id_ for id_, count in counts.items() if count > 1}
    if duplicates:
        print(f"  [WARN] Duplicate IDs detected: {duplicates}. Suffixing.")

    seen: Counter = Counter()
    result = []
    for id_ in ids:
        if id_ in duplicates:
            seen[id_] += 1
            result.append(f"{id_}_dup{seen[id_]}")
        else:
            result.append(id_)
    return result


def _sanitise_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str, int, float, bool as metadata values.
    Replace None with safe defaults and ensure no other types slip through.
    """
    safe = {}
    for k, v in meta.items():
        if v is None:
            safe[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            safe[k] = v
        else:
            # Coerce anything unexpected to string
            safe[k] = str(v)
    return safe


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def run_ingestion():
    print("=" * 60)
    print("Kutub al-Sittah -- Full Ingestion Pipeline")
    print("=" * 60)

    # --- Initialise embedder (validates API key immediately) ---
    print("\n[INIT] Connecting to embedding model...")
    try:
        embedder = HadithEmbedder()
    except ValueError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)
    print(f"       Model: {embedder.model_name}")

    # --- Initialise ChromaDB ---
    print(f"\n[INIT] Opening ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"       Collection '{CHROMA_COLLECTION_NAME}' ready. "
          f"Existing documents: {collection.count()}")

    # --- Load idempotency tracker ---
    ingested_log = _load_ingested_log()
    print(f"\n[INIT] Previously ingested volumes: {len(ingested_log)}")

    # --- Main loop ---
    grand_total = 0
    failed_volumes = []

    for book_key, book_info in BOOKS.items():
        display_name = book_info["display"]
        num_volumes = book_info["volumes"]

        print(f"\n{'-' * 50}")
        print(f"  Book: {display_name}  ({num_volumes} volumes)")
        print(f"{'-' * 50}")

        book_total = 0

        for vol_num in range(1, num_volumes + 1):
            pdf_path = f"data/{book_key}/vol{vol_num}.pdf"
            source_key = f"{book_key}/vol{vol_num}.pdf"

            # --- Skip already-ingested volumes ---
            if source_key in ingested_log:
                prev_count = ingested_log[source_key]
                print(f"  [SKIP] vol{vol_num} -- already ingested "
                      f"({prev_count} hadiths)")
                book_total += prev_count
                grand_total += prev_count
                continue

            # --- Check PDF exists on disk ---
            if not os.path.exists(pdf_path):
                print(f"  [SKIP] vol{vol_num} — PDF not found: {pdf_path}")
                continue

            print(f"  [LOAD] vol{vol_num} — extracting text...", end="", flush=True)

            # --- Extract text (full PDF, no page restriction) ---
            try:
                raw_text = load_pdf_text(pdf_path, start_page=0, end_page=None)
            except Exception as e:
                print(f" FAILED\n  [ERR]  {e}")
                failed_volumes.append(source_key)
                continue

            print(f" done", end="")

            # --- Chunk into hadiths ---
            print(f", chunking...", end="", flush=True)
            try:
                chunks = chunk_text(raw_text, book_key)
            except Exception as e:
                print(f" FAILED\n  [ERR]  {e}")
                failed_volumes.append(source_key)
                continue

            num_chunks = len(chunks)
            print(f" {num_chunks} hadiths found")

            if num_chunks == 0:
                print(f"  [WARN] vol{vol_num} yielded 0 hadiths — skipping.")
                continue

            # --- Embed (bge-m3 shows its own tqdm progress bar per volume) ---
            print(f"         Embedding {num_chunks} hadiths locally "
                  f"(batch_size=16, CPU)...", flush=True)
            try:
                texts_to_embed = [c["content"] for c in chunks]
                embeddings = embedder.embed_texts(texts_to_embed)
            except Exception as e:
                print(f"  [ERR]  Embedding error: {e}", flush=True)
                print(f"         vol{vol_num} will be retried on next run.")
                failed_volumes.append(source_key)
                continue

            print(f"         Embedding done.", flush=True)

            # --- Build IDs and metadata ---
            ids = [
                f"{book_key}_{vol_num}_{c['hadith_number']}"
                for c in chunks
            ]
            ids = _deduplicate_ids(ids)

            metadatas = []
            for c in chunks:
                meta = {
                    "book": book_key,
                    "book_display": display_name,
                    "volume": vol_num,
                    "hadith_number": c["hadith_number"],
                    "source_file": source_key,
                }
                metadatas.append(_sanitise_metadata(meta))

            documents = [c["content"] for c in chunks]

            # --- Add to ChromaDB ---
            print(f"         Storing in ChromaDB...", end="", flush=True)
            try:
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            except Exception as e:
                print(f" FAILED\n  [ERR]  ChromaDB error: {e}")
                failed_volumes.append(source_key)
                continue

            print(f" done")

            # --- Update idempotency log ---
            ingested_log[source_key] = num_chunks
            _save_ingested_log(ingested_log)

            book_total += num_chunks
            grand_total += num_chunks
            db_total = collection.count()
            print(f"  [DONE] {display_name} vol{vol_num} --"
                  f" {num_chunks} hadiths ingested."
                  f" DB total now: {db_total}", flush=True)

        print(f"  Subtotal for {display_name}: {book_total} hadiths")

    # --- Final summary ---
    print(f"\n{'=' * 60}")
    print(f"INGESTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total hadiths ingested this run : {grand_total}")
    print(f"  Total documents in ChromaDB now : {collection.count()}")
    print(f"  Ingested files log              : {INGESTED_FILES_PATH}")

    if failed_volumes:
        print(f"\n  [WARN] The following volumes FAILED and will be retried")
        print(f"         on the next run:")
        for v in failed_volumes:
            print(f"         - {v}")
    else:
        print(f"\n  All volumes processed successfully.")


if __name__ == "__main__":
    run_ingestion()
