"""
=============================================================================
  Kutub al-Sittah — Clean Re-Ingestion Pipeline (Google Colab + T4 GPU)
=============================================================================

HOW TO USE ON COLAB:
  1. Open Google Colab → Runtime → Change runtime type → T4 GPU
  2. Copy each section (marked with # ══ CELL N ══) into separate Colab cells
  3. Run cells in order
  4. Download the chroma_db_v2.zip at the end
  5. Unzip into your project folder on your laptop

WHAT THIS DOES:
  - Downloads meeAtif/hadith_datasets (33,738 hadiths, all 6 books)
  - Runs 7-step validation, filters ~600 bad rows
  - Embeds ~33,100 clean hadiths using BAAI/bge-m3 on T4 GPU
  - Builds a fresh ChromaDB collection (kutub_al_sittah_v2)
  - Zips it for download

ESTIMATED TIME: ~15-20 minutes total on T4 GPU
=============================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies
# ══════════════════════════════════════════════════════════════════════════════

# !pip install -q datasets sentence-transformers chromadb torch

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2: Imports & GPU Check
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import json
import time
import shutil
import numpy as np
import torch
from collections import Counter

print("=" * 60)
print("  Kutub al-Sittah — Clean Re-Ingestion Pipeline")
print("=" * 60)

# GPU check
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    DEVICE = "cuda"
else:
    print("  WARNING: No GPU detected! This will be SLOW on CPU.")
    print("  Go to Runtime > Change runtime type > T4 GPU")
    DEVICE = "cpu"

print(f"  PyTorch: {torch.__version__}")
print(f"  Device:  {DEVICE}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3: Download Dataset
# ══════════════════════════════════════════════════════════════════════════════

from datasets import load_dataset

print("Downloading meeAtif/hadith_datasets from HuggingFace...")
ds = load_dataset("meeAtif/hadith_datasets")
train = ds["train"]
print(f"  Total rows: {len(train)}")
print(f"  Columns:    {train.column_names}")

# Quick per-book count
book_counts = {}
for row in train:
    book_counts[row["Book"]] = book_counts.get(row["Book"], 0) + 1
for book, count in sorted(book_counts.items(), key=lambda x: -x[1]):
    print(f"    {book:30s}  {count}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4: Configuration
# ══════════════════════════════════════════════════════════════════════════════

# ChromaDB settings (must match your .env on laptop)
CHROMA_DB_PATH = "./chroma_db_v2"
COLLECTION_NAME = "kutub_al_sittah_v2"

# Embedding settings
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 64  # T4 handles this well for bge-m3

# Book slug mapping: HuggingFace names → your system's slugs
# These MUST match what retriever.py uses for book filtering
BOOK_SLUG_MAP = {
    "Sahih al-Bukhari":   "bukhari",
    "Sahih Muslim":       "muslim",
    "Jami` at-Tirmidhi":  "tirmidhi",
    "Sunan Abi Dawud":    "abu_dawud",
    "Sunan an-Nasa'i":    "nasai",
    "Sunan Ibn Majah":    "ibn_majah",
}

# Display name mapping (for book_display metadata)
BOOK_DISPLAY_MAP = {
    "Sahih al-Bukhari":   "Sahih al-Bukhari",
    "Sahih Muslim":       "Sahih Muslim",
    "Jami` at-Tirmidhi":  "Jami at-Tirmidhi",
    "Sunan Abi Dawud":    "Sunan Abu Dawud",
    "Sunan an-Nasa'i":    "Sunan an-Nasai",
    "Sunan Ibn Majah":    "Sunan Ibn Majah",
}

# Implicit Sahih books (grade is empty but all hadiths are considered Sahih)
IMPLICIT_SAHIH_BOOKS = {"Sahih al-Bukhari", "Sahih Muslim"}

print("Configuration set.")
print(f"  ChromaDB path:    {CHROMA_DB_PATH}")
print(f"  Collection:       {COLLECTION_NAME}")
print(f"  Embedding model:  {MODEL_NAME}")
print(f"  Batch size:       {BATCH_SIZE}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5: Validation & Filtering
# ══════════════════════════════════════════════════════════════════════════════

print("Running 7-step validation on all 33,738 rows...")
print()

garble_re = re.compile(r'[~@$#%&*\\|+<>{}^]')  # backtick excluded

narrator_patterns = [
    r'narrated', r'reported', r'said\s*:', r'the\s+prophet',
    r'messenger\s+of\s+all', r'allah.s\s+messenger',
    r'prophet.*said', r'hadith',
]

clean_rows = []
rejected = {"garble": 0, "no_narrator": 0, "too_short": 0, "too_long": 0,
            "commentary": 0, "duplicate": 0, "empty": 0, "arabic_dominated": 0}
seen_refs = set()

for idx in range(len(train)):
    row = train[idx]
    text = row.get("English_Text", "") or ""
    book = row.get("Book", "")
    ref = row.get("Reference", "")
    grade = row.get("Grade", "") or ""

    # Check 1: Empty
    if not text.strip():
        rejected["empty"] += 1
        continue

    # Check 2: Too short (<50 chars) — stubs like "Another chain with similar"
    if len(text) < 50:
        rejected["too_short"] += 1
        continue

    # Check 3: Duplicate reference URL
    if ref in seen_refs:
        rejected["duplicate"] += 1
        continue
    seen_refs.add(ref)

    # Check 4: Garble ratio > 1%
    if len(text) > 0:
        garble_count = len(garble_re.findall(text))
        if garble_count / len(text) > 0.01:
            rejected["garble"] += 1
            continue

    # Check 5: No narrator chain (for texts > 100 chars)
    has_narrator = any(re.search(p, text, re.IGNORECASE) for p in narrator_patterns)
    if not has_narrator and len(text) > 100:
        rejected["no_narrator"] += 1
        continue

    # Check 6: Commentary dominated (>70% bracketed text)
    bracket_matches = re.findall(r'\[.*?\]', text, re.DOTALL)
    bracket_len = sum(len(m) for m in bracket_matches)
    if len(text) > 0 and bracket_len / len(text) > 0.70:
        rejected["commentary"] += 1
        continue

    # Check 7: Arabic-dominated English field
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    if len(text) > 0 and arabic_chars / len(text) > 0.50:
        rejected["arabic_dominated"] += 1
        continue

    # ── Passed all checks — prepare clean row ──

    # Fix grade for Bukhari/Muslim
    effective_grade = grade
    if (not grade.strip()) and book in IMPLICIT_SAHIH_BOOKS:
        effective_grade = "Sahih"

    # Extract hadith number from reference URL
    # e.g. "https://sunnah.com/bukhari:3722" → "3722"
    hadith_num = ""
    if ref:
        match = re.search(r':(\w+)$', ref)
        if match:
            hadith_num = match.group(1)

    # Extract volume/book number from in-book reference
    # e.g. "Book 62, Hadith 70" → volume="62"
    in_book_ref = row.get("In-book reference", "") or ""
    volume = ""
    vol_match = re.search(r'Book\s+(\d+)', in_book_ref)
    if vol_match:
        volume = vol_match.group(1)

    # Build slug
    book_slug = BOOK_SLUG_MAP.get(book, "unknown")
    book_display = BOOK_DISPLAY_MAP.get(book, book)

    # Build ChromaDB ID
    chroma_id = f"{book_slug}_{hadith_num}" if hadith_num else f"{book_slug}_{idx}"

    clean_rows.append({
        "id": chroma_id,
        "document": text,
        "metadata": {
            "book": book_slug,
            "book_display": book_display,
            "volume": volume,
            "hadith_number": hadith_num,
            "chapter_number": row.get("Chapter_Number", 0),
            "chapter_title": (row.get("Chapter_Title_English", "") or "").replace("Chapter: ", ""),
            "grade": effective_grade,
            "reference_url": ref,
            "arabic_text": (row.get("Arabic_Text", "") or "")[:2000],  # Cap at 2000 chars for ChromaDB
            "source": "meeAtif/hadith_datasets",
        }
    })

print(f"Validation complete!")
print(f"  PASSED: {len(clean_rows)}")
print(f"  REJECTED: {sum(rejected.values())}")
print(f"  Breakdown:")
for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"    {reason:25s}  {count}")
print()

# Dedup IDs (in case different hadith numbers collide)
id_counts = Counter(r["id"] for r in clean_rows)
dupes = {id_ for id_, c in id_counts.items() if c > 1}
if dupes:
    print(f"  Deduplicating {len(dupes)} colliding IDs...")
    seen_ids = Counter()
    for row in clean_rows:
        if row["id"] in dupes:
            seen_ids[row["id"]] += 1
            row["id"] = f"{row['id']}_v{seen_ids[row['id']]}"
    print(f"  Done. All IDs are now unique.")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6: Load Embedding Model on GPU
# ══════════════════════════════════════════════════════════════════════════════

from sentence_transformers import SentenceTransformer
import gc

print(f"Loading {MODEL_NAME} on {DEVICE}...")
# Clear any residual tensors before loading the model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

t0 = time.time()
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
# Limit max sequence length to 512 tokens to prevent huge VRAM spikes on rare massive texts
# 512 tokens is ~2000+ characters, which is perfect for 99.9% of hadith lengths.
model.max_seq_length = 512
dim = model.get_embedding_dimension()
print(f"  Model loaded in {time.time()-t0:.1f}s")
print(f"  Dimension: {dim}")
print(f"  Max Sequence Length set to: {model.max_seq_length}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7: Embed All Documents in Safe Chunks
# ══════════════════════════════════════════════════════════════════════════════

all_texts = [r["document"] for r in clean_rows]
total = len(all_texts)

print(f"Embedding {total} hadiths on {DEVICE} (batch_size={BATCH_SIZE})...")
print(f"  Estimated time on T4: ~10 minutes")
print()

# Chunk-based embedding to prevent memory fragmentation and spikes
EMBED_CHUNK_SIZE = 5000
all_embeddings_list = []

t0 = time.time()
for chunk_start in range(0, total, EMBED_CHUNK_SIZE):
    chunk_end = min(chunk_start + EMBED_CHUNK_SIZE, total)
    chunk_texts = all_texts[chunk_start:chunk_end]
    
    print(f"  Processing chunk {chunk_start // EMBED_CHUNK_SIZE + 1} ({chunk_start} to {chunk_end})...")
    
    # Explicit garbage collection and cache flushing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    all_embeddings_list.append(chunk_embeddings)

# Stack all chunks back together into one numpy array
all_embeddings = np.vstack(all_embeddings_list)
elapsed = time.time() - t0

print(f"\n  Embedding complete!")
print(f"  Shape: {all_embeddings.shape}")
print(f"  Time:  {elapsed:.1f}s ({total/elapsed:.0f} hadiths/sec)")
print(f"  Norm check (first 3): {[f'{np.linalg.norm(all_embeddings[i]):.4f}' for i in range(3)]}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 8: Build ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

import chromadb

# Remove old DB if exists
if os.path.exists(CHROMA_DB_PATH):
    shutil.rmtree(CHROMA_DB_PATH)
    print(f"  Removed old {CHROMA_DB_PATH}")

print(f"Creating ChromaDB at {CHROMA_DB_PATH}...")
client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=chromadb.config.Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
print(f"  Collection '{COLLECTION_NAME}' created.")
print()

# Insert in batches (ChromaDB recommends batches of ~5000)
CHROMA_BATCH = 5000
total_inserted = 0

print(f"Inserting {total} documents in batches of {CHROMA_BATCH}...")
t0 = time.time()

for start in range(0, total, CHROMA_BATCH):
    end = min(start + CHROMA_BATCH, total)
    batch = clean_rows[start:end]

    batch_ids = [r["id"] for r in batch]
    batch_docs = [r["document"] for r in batch]
    batch_metas = [r["metadata"] for r in batch]
    batch_embeds = all_embeddings[start:end].tolist()

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        embeddings=batch_embeds,
        metadatas=batch_metas,
    )

    total_inserted += len(batch)
    pct = total_inserted / total * 100
    print(f"  Inserted {total_inserted}/{total} ({pct:.1f}%)")

elapsed = time.time() - t0
print(f"\n  ChromaDB build complete!")
print(f"  Total documents: {collection.count()}")
print(f"  Time: {elapsed:.1f}s")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 9: Verification — Test Queries
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("VERIFICATION: Test Queries Against New Collection")
print("=" * 60)

test_queries = [
    "What did the Prophet say about prayer?",
    "Hadith about fasting in Ramadan",
    "Rights of neighbors in Islam",
    "Patience and reward from Allah",
    "Sacrifice of animals on Eid ul Adha",
]

for query in test_queries:
    # Embed query (same way HF API would)
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\n  Q: \"{query}\"")
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            dist = results["distances"][0][i]
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i][:100]
            print(f"    [{i+1}] dist={dist:.4f} | {meta.get('book_display','?')} #{meta.get('hadith_number','?')}")
            print(f"        {doc}...")
    else:
        print("    No results found!")

print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 10: Per-Book Stats
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FINAL STATS: Per-Book Document Counts in ChromaDB")
print("=" * 60)

for book_name, slug in BOOK_SLUG_MAP.items():
    count = len(collection.get(where={"book": slug})["ids"])
    print(f"  {book_name:30s}  {count}")

print(f"  {'TOTAL':30s}  {collection.count()}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 11: Zip for Download
# ══════════════════════════════════════════════════════════════════════════════

# Close ChromaDB client first to flush all writes
del collection
del client

ZIP_NAME = "chroma_db_v2"
print(f"Zipping {CHROMA_DB_PATH} → {ZIP_NAME}.zip ...")
shutil.make_archive(ZIP_NAME, 'zip', '.', CHROMA_DB_PATH)
zip_size = os.path.getsize(f"{ZIP_NAME}.zip") / 1e6
print(f"  Created {ZIP_NAME}.zip ({zip_size:.1f} MB)")
print()

# Colab-specific: trigger download
try:
    from google.colab import files
    print("Downloading to your machine...")
    files.download(f"{ZIP_NAME}.zip")
except ImportError:
    print(f"  Not running in Colab. Zip is at: {os.path.abspath(ZIP_NAME)}.zip")

print()
print("=" * 60)
print("ALL DONE!")
print("=" * 60)
print("""
NEXT STEPS ON YOUR LAPTOP:
  1. Unzip chroma_db_v2.zip into your project folder:
       D:\\Ahadees (RAG) Project\\chroma_db_v2\\

  2. Update your .env file:
       CHROMA_COLLECTION_NAME=kutub_al_sittah_v2
       CHROMA_DB_PATH=./chroma_db_v2

  3. Run: venv\\Scripts\\python.exe chat.py

  4. Test it! If it works, you can delete the old chroma_db folder.
     If not, just revert the .env to use the old collection.
""")
