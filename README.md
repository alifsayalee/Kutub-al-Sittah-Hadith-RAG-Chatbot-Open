# Kutub al-Sittah RAG Pipeline

A robust, fully local Retrieval-Augmented Generation (RAG) pipeline designed to extract, chunk, and embed the six canonical hadith collections (Kutub al-Sittah) in English. 

The pipeline parses raw PDFs, intelligently chunks text into individual hadiths (filtering out commentary and editor notes), and embeds them using the `BAAI/bge-m3` model via `sentence-transformers`. All embeddings are stored locally in a ChromaDB vector database for blazing-fast semantic search.

## Data Source

All of the raw English PDF volumes used in this project are publicly available to download. They were downloaded from [https://www.ahlesunnatpak.com/](https://www.ahlesunnatpak.com/).

Because the raw PDFs are massive, they are actively excluded from this repository. To run the pipeline yourself, you must download the PDFs and place them in the `data/` directory according to the folder structure expected by the loader.

### Corpus Statistics

The project processes **37 volumes** across the 6 canonical books, resulting in exactly **31,414 valid hadiths** stored in the database.

| Book               | Volumes | Total Hadiths | Coverage Notes |
|--------------------|---------|---------------|----------------|
| Sahih al-Bukhari   | 9       | 7,104         | Ends at #7563  |
| Sahih Muslim       | 7       | 6,727         | Ends at #7563  |
| Sunan Abu Dawud    | 5       | 5,054         | Ends at #5274  |
| Jami at-Tirmidhi   | 6       | 3,541         | Ends at #3956  |
| Sunan an-Nasai     | 5       | 4,876         | (5 of 7 vols)  |
| Sunan Ibn Majah    | 5       | 4,112         | Ends at #4341  |
| **TOTAL**          | **37**  | **31,414**    | **~91%**       |

*(Note: Sunan an-Nasai is currently missing 2 volumes in the English translation dataset).*

## Architecture

- **Extraction**: PyMuPDF (`fitz`) for raw text extraction.
- **Chunking**: A custom, state-machine-based chunking logic that handles varied regex formats (`[N]`, `N.`, etc.) and filters non-hadith text using keyword heuristics.
- **Embedding**: `BAAI/bge-m3` running 100% locally on CPU via `sentence-transformers` (1024-dimension vectors). Memory-throttled to safely run on 8GB RAM systems.
- **Vector DB**: `ChromaDB` running persistently on the local disk.

## How to Run

1. **Install dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Data**: Create a `data/` folder and organize the PDFs by book (e.g. `data/bukhari/vol1.pdf`).
3. **Run Ingestion**:
   ```bash
   python -m ingestion.ingest
   ```
   *The first run will automatically download the ~2.2GB BAAI model to your local Hugging Face cache. The ingestion script is highly robust and idempotent—you can stop it (Ctrl+C) at any time and it will instantly resume from where it left off on the next run.*
