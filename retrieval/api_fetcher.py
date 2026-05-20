"""
retrieval/api_fetcher.py

Fetches clean, proofread Hadith translations on-demand from public APIs,
serving as a pristine text source for the `/view` command.

Primary source: fawazahmed0/hadith-api (jsDelivr CDN)
Fallback source: hadithapi.pages.dev (specifically for Sahih Muslim,
                 which has sparse coverage on the primary API)
"""
import urllib.request
import json
import logging
from typing import Optional

# Setup lightweight logger
logger = logging.getLogger(__name__)

# Primary API: fawazahmed0/hadith-api (CDN-hosted, fast, no auth)
BOOK_EDITION_MAPPING = {
    "bukhari": "eng-bukhari",
    "muslim": "eng-muslim",
    "abu_dawud": "eng-abudawud",
    "tirmidhi": "eng-tirmidhi",
    "nasai": "eng-nasai",
    "ibn_majah": "eng-ibnmajah"
}

# Fallback API: hadithapi.pages.dev collection slugs
# Used when the primary API returns empty text (e.g. Sahih Muslim)
FALLBACK_COLLECTION_MAPPING = {
    "bukhari": "bukhari",
    "muslim": "muslim",
    "abu_dawud": "abudawud",
    "tirmidhi": "tirmidhi",
    "nasai": "nasai",
    "ibn_majah": "ibnmajah"
}


def _fetch_from_primary(edition: str, clean_num: str) -> Optional[str]:
    """Try fawazahmed0/hadith-api (jsDelivr CDN)."""
    url = f"https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{edition}/{clean_num}.json"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Kutub-al-Sittah-RAG-Chatbot"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if "hadiths" in data and len(data["hadiths"]) > 0:
                    hadith_text = data["hadiths"][0].get("text")
                    if hadith_text and hadith_text.strip():
                        return hadith_text.strip()
    except Exception as e:
        logger.debug(f"Primary API failed for {edition}/{clean_num}: {e}")
    return None


def _fetch_from_fallback(collection: str, clean_num: str) -> Optional[str]:
    """Try hadithapi.pages.dev as secondary source."""
    url = f"https://hadithapi.pages.dev/api/{collection}/{clean_num}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Kutub-al-Sittah-RAG-Chatbot"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                hadith_text = data.get("hadith_english", "")
                if hadith_text and hadith_text.strip():
                    return hadith_text.strip()
    except Exception as e:
        logger.debug(f"Fallback API failed for {collection}/{clean_num}: {e}")
    return None


def fetch_clean_hadith(book_key: str, hadith_number) -> Optional[str]:
    """
    Fetches clean English translation of a Hadith by book and number.
    
    Tries the primary API first (fawazahmed0/hadith-api). If that returns
    empty text (common for Sahih Muslim), falls back to hadithapi.pages.dev.
    Returns None if all sources fail — the caller should show local text.
    """
    if book_key not in BOOK_EDITION_MAPPING:
        logger.debug(f"Book key '{book_key}' not mapped to any Hadith API.")
        return None
    
    # Coerce to string for type safety (ChromaDB stores str, but guard anyway)
    hadith_number = str(hadith_number)
    
    # Clean Hadith number: strip duplicate suffixes (e.g. "12_dup1" -> "12")
    clean_num = hadith_number.split('_')[0].strip()
    
    # Attempt 1: Primary API
    edition = BOOK_EDITION_MAPPING[book_key]
    text = _fetch_from_primary(edition, clean_num)
    if text:
        return text
    
    # Attempt 2: Fallback API
    collection = FALLBACK_COLLECTION_MAPPING.get(book_key)
    if collection:
        text = _fetch_from_fallback(collection, clean_num)
        if text:
            return text
    
    return None
