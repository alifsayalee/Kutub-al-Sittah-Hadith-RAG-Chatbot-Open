import csv
import os
from ingestion.loader import load_pdf_text
from ingestion.chunker import chunk_text

BOOKS = {
    "bukhari": "data/bukhari/vol1.pdf",
    "muslim": "data/muslim/vol1.pdf",
    "abu_dawud": "data/abu_dawud/vol1.pdf",
    "tirmidhi": "data/tirmidhi/vol1.pdf",
    "nasai": "data/nasai/vol1.pdf",
    "ibn_majah": "data/ibn_majah/vol1.pdf",
}

def main():
    os.makedirs("chunk_validation", exist_ok=True)
    
    total_chunks = 0
    
    for book_name, pdf_path in BOOKS.items():
        if not os.path.exists(pdf_path):
            print(f"[SKIP] {pdf_path} not found")
            continue
            
        print(f"Loading text for {book_name}...")
        
        # In a real scenario we'd do the whole book. For fast validation,
        # we can limit to first 150 pages, but to test accuracy we should 
        # test the full volume.
        # Given it's a test script, let's load the whole PDF (or at least up to page 300)
        # to ensure the chunker is working at scale.
        text = load_pdf_text(pdf_path, start_page=10, end_page=300)
        
        print(f"Chunking {book_name}...")
        chunks = chunk_text(text, book_name)
        
        num_chunks = len(chunks)
        total_chunks += num_chunks
        
        print(f"[DONE] {book_name}: found {num_chunks} chunks.")
        
        csv_path = f"chunk_validation/{book_name}_chunks.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Hadith Number", "Preview (first 150 chars)"])
            for chunk in chunks:
                preview = chunk["content"][:150].replace("\n", " ")
                writer.writerow([chunk["hadith_number"], preview])
                
    print(f"\nTotal chunks found across tested volumes: {total_chunks}")
    print("Please review the CSV files in chunk_validation/ to ensure boundaries are correct.")

if __name__ == "__main__":
    main()
