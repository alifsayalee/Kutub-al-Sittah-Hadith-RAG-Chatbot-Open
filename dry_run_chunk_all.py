"""
Dry-run: chunk ALL 37 volumes without calling the embedding API.
Outputs per-volume counts, flags anomalies, and saves a summary CSV.
"""

import os
import csv
import time
from ingestion.loader import load_pdf_text
from ingestion.chunker import chunk_text

BOOKS = {
    "bukhari":   {"display": "Sahih al-Bukhari",  "volumes": 9},
    "muslim":    {"display": "Sahih Muslim",        "volumes": 7},
    "abu_dawud": {"display": "Sunan Abu Dawud",     "volumes": 5},
    "tirmidhi":  {"display": "Jami at-Tirmidhi",   "volumes": 6},
    "nasai":     {"display": "Sunan an-Nasai",      "volumes": 5},
    "ibn_majah": {"display": "Sunan Ibn Majah",     "volumes": 5},
}


def main():
    os.makedirs("chunk_validation", exist_ok=True)
    summary_rows = []
    grand_total = 0
    warnings = []

    print("=" * 65)
    print("  DRY RUN — Chunking all 37 volumes (no API calls)")
    print("=" * 65)

    for book_key, info in BOOKS.items():
        display = info["display"]
        num_vols = info["volumes"]
        book_total = 0

        print(f"\n{'-' * 55}")
        print(f"  {display}  ({num_vols} volumes)")
        print(f"{'-' * 55}")

        for vol in range(1, num_vols + 1):
            pdf_path = f"data/{book_key}/vol{vol}.pdf"

            if not os.path.exists(pdf_path):
                msg = f"  vol{vol}: PDF NOT FOUND — {pdf_path}"
                print(msg)
                warnings.append(msg)
                summary_rows.append([book_key, display, vol, "MISSING", "", "", ""])
                continue

            t0 = time.time()
            # Full PDF, no page restriction
            raw_text = load_pdf_text(pdf_path, start_page=0, end_page=None)
            t_load = time.time() - t0

            t0 = time.time()
            chunks = chunk_text(raw_text, book_key)
            t_chunk = time.time() - t0

            n = len(chunks)
            book_total += n
            grand_total += n

            # Grab first and last hadith numbers for sanity check
            first_num = chunks[0]["hadith_number"] if chunks else "—"
            last_num = chunks[-1]["hadith_number"] if chunks else "—"

            # Flag anomalies
            flag = ""
            if n == 0:
                flag = "[!] ZERO HADITHS"
                warnings.append(f"{display} vol{vol}: 0 hadiths extracted!")
            elif n < 50:
                flag = "[!] suspiciously low"
                warnings.append(f"{display} vol{vol}: only {n} hadiths")

            print(f"  vol{vol}: {n:>5} hadiths  "
                  f"(#{first_num}..#{last_num})  "
                  f"[load {t_load:.1f}s, chunk {t_chunk:.1f}s]"
                  f"  {flag}")

            summary_rows.append([
                book_key, display, vol, n, first_num, last_num,
                f"{t_load + t_chunk:.1f}s"
            ])

        print(f"  Book total: {book_total}")

    # Save summary CSV
    csv_path = "chunk_validation/full_dry_run_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Book Key", "Display Name", "Volume",
            "Chunk Count", "First Hadith #", "Last Hadith #", "Time"
        ])
        writer.writerows(summary_rows)

    # Final report
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total hadiths across all volumes: {grand_total}")
    print(f"  Summary saved to: {csv_path}")

    if warnings:
        print(f"\n  [!] WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    {w}")
    else:
        print(f"\n  [OK] No warnings — all volumes chunked successfully.")


if __name__ == "__main__":
    main()
