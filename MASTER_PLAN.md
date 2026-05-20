# Kutub al-Sittah RAG Pipeline — Master Implementation Plan

> This is the PERMANENT master plan for the project. It must NEVER be overwritten.
> Updates are appended or edited in-place. A copy also exists as an artifact.
> Last updated: 2026-05-13

---

## Project Overview

Build a full Retrieval-Augmented Generation (RAG) pipeline for the six canonical
hadith collections (Kutub al-Sittah) in English. The system ingests 37 PDF volumes,
chunks them into individual hadiths, embeds them as vectors, stores them in ChromaDB,
and provides a search/retrieval interface powered by an LLM.

**Data scope:** English-only (Arabic extraction not feasible due to OCR/compute limits).
**Embedding model:** BAAI/bge-m3 (run locally via sentence-transformers, no paid API).
**Vector DB:** ChromaDB (local, single collection, metadata-filtered per book).
**LLM:** Google Gemini API (free tier) for the Generation step of RAG.

---

## Phase 1 — PDF Extraction & Chunking [COMPLETED]

### What was done
- **Loader** (`ingestion/loader.py`): Extracts raw text from PDFs using PyMuPDF.
- **Chunker** (`ingestion/chunker.py`): State-machine based chunker with:
  - Per-book regex patterns in `BOOK_CONFIGS`
  - Sequential number tracking to distinguish real hadiths from editorial numbering
  - Keyword filtering (first 200 chars must contain narration keywords)
  - `RESET_THRESHOLD` to recover from long prefaces with numbered lists
  - Text cleanup preserving paragraph structure
- **Muslim regex fix**: Changed from `^\[(\d{1,5})\]\s+\d{1,5}\s+-` to `^\[(\d{1,5})\]`
  to catch sub-narration entries with `( ... )` format. Recovered 1,733 additional hadiths.

### Validation Results (full dry run, all 37 volumes)

| Book               | Volumes | Chunks | Last # | Coverage |
|--------------------|---------|--------|--------|----------|
| Sahih al-Bukhari   | 9       | 7,104  | #7563  | ~94%     |
| Sahih Muslim       | 7       | 6,727  | #7563  | ~89%     |
| Sunan Abu Dawud    | 5       | 5,054  | #5274  | ~96%     |
| Jami at-Tirmidhi   | 6       | 3,541  | #3956  | ~90%     |
| Sunan an-Nasai     | 5       | 4,876  | #4987  | ~85%*    |
| Sunan Ibn Majah    | 5       | 4,112  | #4341  | ~95%     |
| **TOTAL**          | **37**  |**31,414**| —    | **~91%** |

*Nasai: only 5 of 7 volumes available in English.

### Key files
- `ingestion/loader.py` — PDF text extraction
- `ingestion/chunker.py` — State-machine hadith chunker
- `chunk_validation/full_dry_run_summary.csv` — Validation data
- `dry_run_chunk_all.py` — Dry run script (used for validation)

---

## Phase 2 — Embedding & ChromaDB Ingestion [COMPLETED]

### What was done
- **Embedder** (`ingestion/embedder.py`): Originally built with Google's `google-genai`
  SDK for `gemini-embedding-001`. Smoke test passed (3072-dim vectors returned).
  Google free tier rate limits (100 RPM, 1000 RPD) caused all embedding calls to fail.
  Switched to local BAAI/bge-m3 via sentence-transformers (1024-dim vectors, CPU-only).
- **Ingest orchestrator** (`ingestion/ingest.py`): Full pipeline with:
  - Idempotency via `ingestion/ingested_files.json`
  - Per-volume error handling (failed volume logged, doesn't abort entire run)
  - Metadata sanitisation for ChromaDB compatibility
  - Deduplication of IDs within volumes
- **Throttling for 8GB RAM**: batch_size=4, PyTorch limited to 2 threads, all
  telemetry disabled (HuggingFace + ChromaDB).
- **Full ingestion completed**: 31,414 hadiths embedded and stored in ChromaDB.
- **Verification passed**: Semantic search test confirmed correct retrieval.
  Metadata filtering by book confirmed working.

### ChromaDB Architecture
- Single collection: `kutub_al_sittah`
- All 6 books in one collection, differentiated by metadata:
  ```python
  {"book": "bukhari", "book_display": "Sahih al-Bukhari", "volume": 1, ...}
  ```
- At query time, filter with `where={"book": "bukhari"}` or query all books at once
- cosine similarity metric (`hnsw:space: cosine`)

### Key files
- `ingestion/embedder.py` — Local BAAI/bge-m3 embedding wrapper
- `ingestion/ingest.py` — Ingestion orchestrator
- `ingestion/ingested_files.json` — Idempotency tracker (created at runtime)
- `.env` — Configuration (paths, model names)

---

## Phase 3 — Retrieval & RAG Generation [COMPLETED]

This is the core RAG phase. The system retrieves relevant hadiths from ChromaDB
and then passes them as context to a Gemini LLM to generate a coherent answer.

### Step 3a: Retrieval Module (`retrieval/retriever.py`)
1. Takes a user's natural language question
2. Embeds it using the same BAAI/bge-m3 model (query mode, done through HF API)
3. Searches ChromaDB for the top-K most similar hadiths
4. Returns results with metadata (book, volume, hadith number, text)

### Step 3b: LLM Generation (`retrieval/generator.py`)
1. Takes the user's original question + the retrieved hadiths as context
2. Sends them to **Google Gemini API** (free tier) as a prompt
3. The LLM generates a 2-3 paragraph answer grounded in the retrieved hadiths
4. Returns the LLM's answer alongside the source hadiths and their metadata

### Required Output Structure
The final output shown to the user must follow this exact format:
1. **LLM Answer** (2-3 paragraphs): Generated by Gemini from the query + retrieved
   hadith context. This is the synthesised, readable answer.
2. **Referenced Hadiths** (exact English text): The actual hadith texts that the
   LLM used to form its answer. Printed in full so the user can read the originals.
3. **Metadata**: For each referenced hadith, print the book name, volume number,
   and hadith number so anyone can independently verify or look up the hadith.

### Design decisions (planned)
- Use same `HadithEmbedder.embed_query()` method for query embedding
- Support filtering by book (`where={"book": "bukhari"}`)
- Configurable `TOP_K` and `SIMILARITY_THRESHOLD` from `.env`
- Gemini API key stored in `.env` as `GOOGLE_API_KEY`
- Use `google-genai` SDK for the LLM call (same package, different model)

---

## Phase 4 — CLI Application [COMPLETED]

### Goal
Build an interactive terminal interface where the user can:
1. Type a question in natural language
2. Optionally filter by book
3. See the full RAG output in the structured format:
   - LLM-generated answer (2-3 paragraphs)
   - Exact hadith texts referenced
   - Metadata (book, volume, hadith number) for each referenced hadith
4. Continue asking questions in a loop

---

## Environment & Dependencies

### Current packages (requirements.txt)
- `pymupdf` — PDF text extraction
- `chromadb` — Vector database
- `python-dotenv` — Environment variable management
- `sentence-transformers` — Local embedding with BAAI/bge-m3
- `google-genai` — Google Gemini API SDK for the LLM generation step
- `pydantic` — Data validation

### Environment variables (.env)
```
GOOGLE_API_KEY=<your_gemini_api_key>   # For LLM generation (Gemini free tier)
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=kutub_al_sittah
TOP_K=5
SIMILARITY_THRESHOLD=0.65
HF_HUB_DISABLE_TELEMETRY=1
```

### System constraints
- **RAM:** 8GB — batch sizes must be kept small (8-16 texts)
- **GPU:** Not available — CPU-only inference
- **OS:** Windows

---

## Progress Log

| Date       | What was done                                                    |
|------------|------------------------------------------------------------------|
| 2026-05-12 | Phase 1 complete: loader, chunker, validation across 37 volumes  |
| 2026-05-12 | Muslim regex fix: recovered 1,733 hadiths (4,994 -> 6,727)      |
| 2026-05-12 | Phase 2 partial: embedder + ingest.py written, smoke test passed |
| 2026-05-12 | Google API rate limit hit; decided to switch to local BAAI/bge-m3|
| 2026-05-13 | Master plan restored and saved as physical file in project root  |
| 2026-05-13 | Phase 2 COMPLETED: 31,414 hadiths embedded & stored in ChromaDB  |
| 2026-05-13 | Semantic search + metadata filtering verified working            |
| 2026-05-13 | Plan updated: RAG/LLM generation details added (Gemini free tier)|
| 2026-05-20 | Phase 3 COMPLETED: Separate retrieval & generation modules verified working |
| 2026-05-20 | Phase 4 COMPLETED: Beautiful interactive CLI app (`chat.py`) built using `rich` |
