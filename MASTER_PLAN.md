# Kutub al-Sittah RAG Pipeline — Master Implementation Plan

> This is the PERMANENT master plan for the project. It must NEVER be overwritten.
> Updates are appended or edited in-place. A copy also exists as an artifact.
> Last updated: 2026-05-22

---

## Project Overview

Build a full Retrieval-Augmented Generation (RAG) pipeline for the six canonical
hadith collections (Kutub al-Sittah) in English. The system downloads a clean,
pre-structured dataset from HuggingFace, validates it, embeds all hadiths using
BAAI/bge-m3 on a Google Colab T4 GPU, and stores them in a ChromaDB vector database.
At runtime, users query through a rich CLI, and the system retrieves relevant hadiths
and generates grounded answers via an LLM.

**Data source:** `meeAtif/hadith_datasets` on HuggingFace (scraped from sunnah.com, MIT licensed).
**Embedding model:** BAAI/bge-m3 (1024-dim, embedded on Colab T4 GPU, queried via HF Inference API).
**Vector DB:** ChromaDB (local, single collection `kutub_al_sittah_v2`, metadata-filtered per book).
**LLM:** OpenRouter API (Gemini 2.5 Flash) for the Generation step of RAG.

---

## Phase 1 — Data Preprocessing & Validation [COMPLETED]

### Data Source
Previously, data was extracted from 37 scanned PDF volumes using PyMuPDF OCR, which
introduced significant garble and noise (~53% of queries returned zero results).

The pipeline was migrated to use `meeAtif/hadith_datasets` from HuggingFace — a clean,
digital-native dataset scraped directly from sunnah.com. Each row is one hadith with
structured fields (English text, Arabic text, book, grade, reference URL, etc.).

### 7-Step Validation Pipeline
Every hadith passes through validation before embedding:

1. **Non-empty** — English text must exist
2. **Length >= 50 chars** — filters stubs like "Another chain with similar"
3. **No duplicate reference URLs** — deduplicates across the dataset
4. **Garble ratio <= 1%** — checks for special characters (backtick excluded — sunnah.com transliteration)
5. **Narrator chain present** — relaxed: `narrated` anywhere (for texts >100 chars)
6. **Not commentary-dominated** — rejects if >70% is bracketed text
7. **Not Arabic-dominated** — rejects if >50% of English field is Arabic characters

### Validation Results (full dataset, all 33,738 rows)

| Metric | Value |
|---|---|
| **Total rows** | 33,738 |
| **PASSED** | 33,141 (98.2%) |
| **REJECTED** | 597 (1.8%) |
| Rejection: too short | 181 |
| Rejection: no narrator | 176 |
| Rejection: duplicate | 146 |
| Rejection: commentary | 88 |
| Rejection: garble | 6 |

### Key files
- `scripts/colab_reingest.py` — Full pipeline (download → validate → embed → ChromaDB)

---

## Phase 2 — Embedding & ChromaDB Ingestion [COMPLETED]

### What was done
- **Embedding on Google Colab T4 GPU** using `BAAI/bge-m3` (1024-dim vectors).
- Chunked processing (5,000 hadiths at a time) with `gc.collect()` + `torch.cuda.empty_cache()`
  between chunks to prevent GPU memory fragmentation.
- `max_seq_length = 1024` tokens to cap VRAM spikes on rare very long texts.
- `normalize_embeddings=True` for cosine similarity.
- English text only embedded — Arabic stored in metadata for display.
- Total embedding time: ~10 minutes on T4 GPU.

### ChromaDB Architecture
- Single collection: `kutub_al_sittah_v2`
- All 6 books in one collection, differentiated by metadata:
  ```python
  {"book": "bukhari", "book_display": "Sahih al-Bukhari", "volume": "62", ...}
  ```
- At query time, filter with `where={"book": "bukhari"}` or query all books at once
- Cosine similarity metric (`hnsw:space: cosine`)
- **33,279 hadiths** successfully embedded and stored

### Per-Book Document Counts

| Book | Documents | Percentage |
|---|---|---|
| **Sahih Muslim** | 7,335 | 22.0% |
| **Sahih al-Bukhari** | 7,241 | 21.8% |
| **Sunan an-Nasai** | 5,677 | 17.1% |
| **Sunan Abu Dawud** | 4,997 | 15.0% |
| **Sunan Ibn Majah** | 4,059 | 12.2% |
| **Jami at-Tirmidhi** | 3,970 | 11.9% |
| **TOTAL** | **33,279** | **100%** |

### Metadata Schema (per hadith)

```python
{
    "book":            "bukhari",                          # Slug for filtering
    "book_display":    "Sahih al-Bukhari",                 # Display name
    "volume":          "62",                               # From In-book reference
    "hadith_number":   "3722",                             # From reference URL
    "chapter_number":  62,                                 # Integer
    "chapter_title":   "Virtues of the Companions",        # English chapter name
    "grade":           "Sahih",                            # Authenticity grading
    "reference_url":   "https://sunnah.com/bukhari:3722",  # Direct sunnah.com link
    "arabic_text":     "...",                               # Arabic text (capped 2000 chars)
    "source":          "meeAtif/hadith_datasets",          # Provenance
}
```

### Key files
- `scripts/colab_reingest.py` — Re-ingestion pipeline (Colab cells)
- `scripts/test_v2_db.py` — Comprehensive DB verification (8-test suite)
- `.env` — Configuration (paths, API keys, collection name)

---

## Phase 3 — Retrieval & RAG Generation [COMPLETED]

This is the core RAG phase. The system retrieves relevant hadiths from ChromaDB
and then passes them as context to an LLM to generate a coherent answer.

### Step 3a: Retrieval Module (`retrieval/retriever.py`)
1. Takes a user's natural language question
2. Embeds it using BAAI/bge-m3 via the HF Inference API (free tier)
3. Searches ChromaDB for the top-K (default 3) most similar hadiths
4. Filters results by cosine distance ceiling (0.42)
5. Returns results with metadata (book, volume, hadith number, grade, reference URL)

### Step 3b: LLM Generation (`retrieval/generator.py`)
1. Takes the user's original question + the retrieved hadiths as context
2. Sends them to **OpenRouter API** (Gemini 2.5 Flash) as a prompt
3. The LLM generates a 2-3 paragraph answer grounded in the retrieved hadiths
4. Returns the LLM's answer alongside the source hadiths and their metadata

### Required Output Structure
The final output shown to the user must follow this exact format:
1. **LLM Answer** (2-3 paragraphs): Generated by the LLM from the query + retrieved
   hadith context. This is the synthesised, readable answer.
2. **Referenced Hadiths** (exact English text): The actual hadith texts that the
   LLM used to form its answer. Printed in full so the user can read the originals.
3. **Metadata**: For each referenced hadith, print the book name, volume number,
   and hadith number so anyone can independently verify or look up the hadith.

### Dual-Layer Safety Guardrails
- **Layer 1 (Distance Shield):** Cosine distance > 0.42 → immediate polite refusal
- **Layer 2 (LLM Sentinel):** LLM outputs `[REFUSAL_SHIELD]` for out-of-scope queries

---

## Phase 4 — CLI Application [COMPLETED]

### Implementation (`chat.py`)
Built an interactive terminal interface using `rich` where the user can:
1. Type a question in natural language
2. Optionally filter by book (0 = all, 1-6 = specific book)
3. See the full RAG output: LLM answer + referenced hadiths + metadata
4. Use `/view <number>` to fetch clean digital text from public APIs
5. Continue asking questions in a loop

### Interactive Commands
- `/filter` — Switch the targeted collection or book mid-session
- `/view <number>` — Fetch pristine digital text via dual-API cascade
- `/clear` — Clear the terminal screen
- `exit` / `quit` — Gracefully close

### /view Command — Dual-API Cascade (`retrieval/api_fetcher.py`)
1. **Primary API:** Fawazahmed's digital Hadith CDN (fast, jsDelivr CDN)
2. **Fallback API:** hadithapi.pages.dev (excellent for Sahih Muslim)
3. **Local Failsafe:** Local database text if offline

---

## Environment & Dependencies

### Current packages (requirements.txt)
- `chromadb` — Vector database
- `python-dotenv` — Environment variable management
- `huggingface-hub` — HF Inference API for query embedding
- `requests` — HTTP calls to OpenRouter API
- `rich` — Beautiful terminal UI
- `pydantic` — Data validation

### Environment variables (.env)
```
HF_API_TOKEN=<your_huggingface_token>      # For query embedding (HF Inference API)
OPEN_ROUTER_API_KEY=<your_openrouter_key>  # For LLM generation
CHROMA_DB_PATH=./chroma_db_v2
CHROMA_COLLECTION_NAME=kutub_al_sittah_v2
TOP_K=3
HF_HUB_DISABLE_TELEMETRY=1
```

---

## Project Structure

```
Ahadees (RAG) Project/
├── chat.py                  # Main CLI application (entry point)
├── retrieval/
│   ├── retriever.py         # Query embedding + ChromaDB search
│   ├── generator.py         # LLM answer generation (OpenRouter)
│   └── api_fetcher.py       # Clean text fetcher for /view command
├── scripts/
│   ├── colab_reingest.py    # Full re-ingestion pipeline (Colab notebook)
│   └── test_v2_db.py        # Comprehensive DB verification suite
├── chroma_db_v2/            # ChromaDB database (33,279 hadiths)
├── .env                     # API keys & configuration
├── requirements.txt         # Python dependencies
├── MASTER_PLAN.md           # This file
└── README.md                # GitHub-facing documentation
```

---

## Progress Log

| Date       | What was done                                                    |
|------------|------------------------------------------------------------------|
| 2026-05-12 | Phase 1 (old): PDF loader, chunker, validation across 37 volumes |
| 2026-05-12 | Phase 2 (old): 31,414 hadiths embedded from PDFs into ChromaDB  |
| 2026-05-13 | Verified semantic search + metadata filtering working            |
| 2026-05-20 | Phase 3: Retrieval & generation modules (HF API + OpenRouter)    |
| 2026-05-20 | Phase 4: Interactive CLI app (chat.py) built using rich          |
| 2026-05-21 | Diagnosed OCR noise problem — 53% query failure rate             |
| 2026-05-21 | Migrated data source: PDFs → HuggingFace meeAtif/hadith_datasets|
| 2026-05-21 | 7-step validation on all 33,738 rows — 98.2% pass rate          |
| 2026-05-21 | Embedding compatibility verified (local vs HF API = identical)   |
| 2026-05-22 | Re-ingestion on Colab T4 GPU — 33,279 hadiths in chroma_db_v2   |
| 2026-05-22 | Full verification: 32/33 tests passed, all queries returning relevant results |
| 2026-05-22 | Project cleanup: removed old PDF pipeline, restructured codebase |
