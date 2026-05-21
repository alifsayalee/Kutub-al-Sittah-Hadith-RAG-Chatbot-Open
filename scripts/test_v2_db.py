import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
r"""
test_v2_db.py  —  Comprehensive ChromaDB v2 Verification
=========================================================
Tests everything chat.py and retriever.py depend on:
  1. Path resolution (detects nested zip issue)
  2. Collection existence & document count
  3. Per-book distribution
  4. Metadata field completeness
  5. Embedding dimensionality & normalization
  6. Live query test via HF Inference API (same as retriever.py)
  7. Distance sanity check
  8. Book filter test (same as chat.py)

Run:  venv\Scripts\python.exe test_v2_db.py
"""

import os
import sys
import re
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "kutub_al_sittah_v2")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

EXPECTED_BOOKS = {"bukhari", "muslim", "abu_dawud", "tirmidhi", "nasai", "ibn_majah"}
EXPECTED_MIN_PER_BOOK = {
    "bukhari": 7000,
    "muslim": 5000,
    "abu_dawud": 4500,
    "tirmidhi": 3500,
    "nasai": 5000,
    "ibn_majah": 4000,
}
EXPECTED_TOTAL_MIN = 32000
EXPECTED_TOTAL_MAX = 34000
REQUIRED_META_FIELDS = ["book", "book_display", "hadith_number", "reference_url", "grade", "source"]

passed = 0
failed = 0
warnings = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  —  {detail}")
        failed += 1

def warn(name, detail):
    global warnings
    print(f"  [WARN] {name}  —  {detail}")
    warnings += 1

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Path Resolution
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Path Resolution")
print("=" * 70)

resolved_path = os.path.abspath(CHROMA_DB_PATH)
print(f"  .env CHROMA_DB_PATH = {CHROMA_DB_PATH}")
print(f"  Resolved to:          {resolved_path}")

# Check for nested zip issue: chroma_db_v2/chroma_db_v2/chroma.sqlite3
nested_path = os.path.join(resolved_path, os.path.basename(resolved_path))
has_nested = os.path.isdir(nested_path) and os.path.exists(os.path.join(nested_path, "chroma.sqlite3"))
direct_sqlite = os.path.exists(os.path.join(resolved_path, "chroma.sqlite3"))

if has_nested and not direct_sqlite:
    print(f"\n  *** NESTED ZIP DETECTED ***")
    print(f"  The zip created: {resolved_path}/{os.path.basename(resolved_path)}/chroma.sqlite3")
    print(f"  But .env points to: {resolved_path}")
    print(f"  FIX: Updating .env to use the nested path...")
    
    # Auto-fix: use the nested path
    CHROMA_DB_PATH = nested_path
    resolved_path = nested_path
    print(f"  Using corrected path: {resolved_path}")
    NEEDS_ENV_FIX = True
else:
    NEEDS_ENV_FIX = False

test("Directory exists", os.path.isdir(resolved_path), f"Path not found: {resolved_path}")
test("chroma.sqlite3 exists", os.path.exists(os.path.join(resolved_path, "chroma.sqlite3")),
     f"No chroma.sqlite3 in {resolved_path}")

sqlite_size = 0
if os.path.exists(os.path.join(resolved_path, "chroma.sqlite3")):
    sqlite_size = os.path.getsize(os.path.join(resolved_path, "chroma.sqlite3")) / 1e6
    print(f"  SQLite size: {sqlite_size:.1f} MB")
    test("SQLite size reasonable (>50 MB)", sqlite_size > 50, f"Only {sqlite_size:.1f} MB — suspiciously small")

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Collection Access & Document Count
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: Collection Access & Document Count")
print("=" * 70)

import chromadb

try:
    client = chromadb.PersistentClient(
        path=resolved_path,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    collections = [c.name for c in client.list_collections()]
    print(f"  Available collections: {collections}")
    test("Collection exists", COLLECTION_NAME in collections,
         f"'{COLLECTION_NAME}' not found. Available: {collections}")
    
    collection = client.get_collection(name=COLLECTION_NAME)
    total_count = collection.count()
    print(f"  Total documents: {total_count}")
    test(f"Total count >= {EXPECTED_TOTAL_MIN}", total_count >= EXPECTED_TOTAL_MIN,
         f"Only {total_count} documents")
    test(f"Total count <= {EXPECTED_TOTAL_MAX}", total_count <= EXPECTED_TOTAL_MAX,
         f"{total_count} documents — more than expected")
except Exception as e:
    print(f"  [FAIL] Could not open ChromaDB: {e}")
    failed += 1
    print(f"\nABORTING — cannot proceed without database access.")
    sys.exit(1)

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Per-Book Distribution
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 3: Per-Book Document Counts")
print("=" * 70)

book_counts = {}
for slug in EXPECTED_BOOKS:
    result = collection.get(where={"book": slug})
    count = len(result["ids"])
    book_counts[slug] = count
    min_expected = EXPECTED_MIN_PER_BOOK.get(slug, 3000)
    test(f"{slug:12s} has >= {min_expected} docs", count >= min_expected, f"Only {count}")

sum_books = sum(book_counts.values())
test("Sum of per-book counts == total", sum_books == total_count,
     f"Sum={sum_books} vs total={total_count}")

print(f"\n  Per-book breakdown:")
for slug in sorted(book_counts, key=lambda x: -book_counts[x]):
    pct = book_counts[slug] / total_count * 100
    print(f"    {slug:15s}  {book_counts[slug]:6d}  ({pct:.1f}%)")
print(f"    {'TOTAL':15s}  {total_count:6d}")

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Metadata Field Completeness (sample 100 random docs)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 4: Metadata Field Completeness (100 random samples)")
print("=" * 70)

import random
sample_result = collection.get(limit=100, include=["metadatas", "documents"])
sample_metas = sample_result["metadatas"]
sample_docs = sample_result["documents"]

for field in REQUIRED_META_FIELDS:
    present = sum(1 for m in sample_metas if m.get(field))
    test(f"'{field}' present in all 100 samples", present == 100,
         f"Only {present}/100 have '{field}'")

# Check book_display values match expected display names
EXPECTED_DISPLAYS = {
    "Sahih al-Bukhari", "Sahih Muslim", "Sunan Abu Dawud",
    "Jami at-Tirmidhi", "Sunan an-Nasai", "Sunan Ibn Majah"
}
found_displays = set(m.get("book_display", "") for m in sample_metas)
for d in found_displays:
    if d and d not in EXPECTED_DISPLAYS:
        warn("Unexpected book_display value", d)

# Check documents are non-empty and reasonable
empty_docs = sum(1 for d in sample_docs if not d or len(d.strip()) < 50)
test("No empty/stub documents in sample", empty_docs == 0,
     f"{empty_docs}/100 are empty or too short")

# Check for garble in sampled documents
garble_re = re.compile(r'[~@$#%&*\\|+<>{}^]')
garbled = 0
for doc in sample_docs:
    if doc and len(doc) > 0:
        ratio = len(garble_re.findall(doc)) / len(doc)
        if ratio > 0.01:
            garbled += 1
test("No garbled documents in sample", garbled == 0,
     f"{garbled}/100 have garble ratio > 1%")

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Embedding Dimensionality & Normalization (sample 10)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 5: Embedding Dimensionality & Normalization")
print("=" * 70)

embed_sample = collection.get(limit=10, include=["embeddings"])
embeddings = embed_sample["embeddings"]

test("Embeddings returned", embeddings is not None and len(embeddings) > 0)

if embeddings is not None and len(embeddings) > 0:
    dim = len(embeddings[0])
    test(f"Dimension is 1024", dim == 1024, f"Got {dim}")

    norms = [np.linalg.norm(e) for e in embeddings]
    all_unit = all(0.99 < n < 1.01 for n in norms)
    test("All embeddings are unit-normalized", all_unit,
         f"Norms: {[f'{n:.4f}' for n in norms]}")

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Live Query via HF Inference API (same path as retriever.py)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 6: Live Query via HF Inference API")
print("=" * 70)

if not HF_API_TOKEN:
    warn("HF_API_TOKEN not set", "Skipping live query test")
else:
    from huggingface_hub import InferenceClient
    hf_client = InferenceClient(model="BAAI/bge-m3", token=HF_API_TOKEN)

    test_queries = [
        ("What did the Prophet say about prayer?", None),
        ("Hadith about fasting in Ramadan", None),
        ("Rights of neighbors in Islam", None),
        ("Patience and reward from Allah", None),
        ("Hadith about honesty from Sahih Bukhari", "bukhari"),
    ]

    for query, book_filter in test_queries:
        try:
            q_vec = hf_client.feature_extraction(query).tolist()
            
            where_clause = {"book": book_filter} if book_filter else None
            results = collection.query(
                query_embeddings=[q_vec],
                n_results=3,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            has_results = bool(results["ids"] and results["ids"][0])
            top_dist = results["distances"][0][0] if has_results else 999
            filter_label = f" [filter={book_filter}]" if book_filter else ""
            
            test(f"Query: '{query[:45]}...'{filter_label} → {len(results['ids'][0]) if has_results else 0} results, dist={top_dist:.4f}",
                 has_results and top_dist < 0.42,
                 f"dist={top_dist:.4f}" if has_results else "no results")

            if has_results:
                meta = results["metadatas"][0][0]
                doc_preview = results["documents"][0][0][:80]
                print(f"         -> [{meta.get('book_display','?')}] #{meta.get('hadith_number','?')}: {doc_preview}...")
                
                # Verify book filter worked
                if book_filter:
                    all_correct_book = all(
                        m.get("book") == book_filter
                        for m in results["metadatas"][0]
                    )
                    test(f"  Book filter '{book_filter}' applied correctly", all_correct_book)
        except Exception as e:
            print(f"  [FAIL] Query failed: {e}")
            failed += 1

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Duplicate ID Check
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 7: Duplicate ID Check (full scan)")
print("=" * 70)

all_ids = collection.get(include=[])["ids"]
unique_ids = set(all_ids)
test("No duplicate IDs", len(all_ids) == len(unique_ids),
     f"{len(all_ids) - len(unique_ids)} duplicates found")

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: .env Configuration Check
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 8: .env Configuration")
print("=" * 70)

env_collection = os.getenv("CHROMA_COLLECTION_NAME", "")
env_path = os.getenv("CHROMA_DB_PATH", "")
test(".env CHROMA_COLLECTION_NAME is kutub_al_sittah_v2",
     env_collection == "kutub_al_sittah_v2",
     f"Got: '{env_collection}'")

if NEEDS_ENV_FIX:
    print(f"\n  *** ACTION REQUIRED ***")
    print(f"  Your .env has:  CHROMA_DB_PATH={env_path}")
    print(f"  But the DB is at: {resolved_path}")
    print(f"  You need to either:")
    print(f"    (a) Move contents of chroma_db_v2/chroma_db_v2/* up to chroma_db_v2/")
    print(f"    (b) Change .env to: CHROMA_DB_PATH=./chroma_db_v2/chroma_db_v2")
    failed += 1
else:
    test(".env CHROMA_DB_PATH resolves correctly", direct_sqlite)

print()

# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
total_tests = passed + failed
print(f"RESULTS:  {passed}/{total_tests} passed,  {failed} failed,  {warnings} warnings")
print("=" * 70)

if failed == 0:
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  ALL TESTS PASSED!  ChromaDB v2 is verified and ready.     ║
  ║                                                            ║
  ║  SAFE to delete old databases:                             ║
  ║    • chroma_db      (old OCR-based collection)             ║
  ║    • chroma_db_test (test collection)                      ║
  ╚══════════════════════════════════════════════════════════════╝
""")
else:
    print(f"\n  WARNING: {failed} test(s) FAILED -- review issues above before proceeding.\n")
