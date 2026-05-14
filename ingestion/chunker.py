import re
from typing import List, Dict

# Regex patterns for detecting potential hadith boundaries per book.
# Each pattern has a capture group for the hadith number.
BOOK_CONFIGS = {
    "bukhari": r"(?m)^(\d{1,5})\.\s+",
    "muslim": r"(?m)^\[(\d{1,5})\]",
    "abu_dawud": r"(?m)^(\d{1,5})\.\s+",
    "tirmidhi": r"(?m)^(\d{1,5})\.\s+",
    "nasai": r"(?m)^(\d{1,5})\.\s+",
    "ibn_majah": r"(?m)^(\d{1,5})\.\s+"
}

# Keywords that strongly indicate a real hadith (checked in first 200 chars)
NARRATION_KEYWORDS = [
    "narrated", "reported", "said:", "said,", "said that",
    "told us", "told me", "told him",
    "it was narrated", "it has been related",
    "it was reported",
    "i heard", "i saw", "i asked",
    "the prophet", "the messenger",
    "allah's messenger",
]


def _has_narration_keyword(text: str) -> bool:
    """Check if the first 200 characters contain a narration keyword."""
    snippet = text[:200].lower()
    return any(kw in snippet for kw in NARRATION_KEYWORDS)


def chunk_text(text: str, book_name: str) -> List[Dict]:
    """
    Splits raw PDF text into individual hadith chunks using a two-pass
    strategy: regex boundary detection + sequential state machine with
    keyword validation.

    Returns a list of dicts: {"hadith_number": str, "content": str}

    The state machine prevents commentary bullet points (1. 2. 3.) and
    introduction numbered lists from being split as separate hadiths.
    A match is accepted as a real hadith boundary if EITHER:
      - Its number is sequential (within a gap tolerance of 5), OR
      - It contains a narration keyword AND its number >= last accepted number
    Everything else is appended to the previous hadith's content.
    """
    if book_name not in BOOK_CONFIGS:
        raise ValueError(f"Unknown book: {book_name}")

    pattern = BOOK_CONFIGS[book_name]

    # re.split with a capture group returns:
    # [pre_text, num_1, chunk_1, num_2, chunk_2, ...]
    parts = re.split(pattern, text)

    if len(parts) < 3:
        # No matches found at all
        return []

    # --- Pass 1: Pair up all (number, content) candidates ---
    candidates = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            num_str = parts[i].strip()
            content = parts[i + 1]
            try:
                num = int(num_str)
            except ValueError:
                continue
            candidates.append({"num": num, "num_str": num_str, "content": content})

    if not candidates:
        return []

    # --- Pass 2: State machine to separate real hadiths from noise ---
    chunks = []
    current_hadith = None
    expected_num = None
    skipped_count = 0  # Track consecutive non-matches for reset logic
    GAP_TOLERANCE = 5  # Allow gaps up to 5 (some hadiths may not extract)
    RESET_THRESHOLD = 15  # Reset state machine after this many consecutive skips

    for cand in candidates:
        num = cand["num"]
        content = cand["content"]
        has_keyword = _has_narration_keyword(content)

        is_sequential = False
        if expected_num is not None:
            # Sequential: the number is within the expected range
            is_sequential = (expected_num <= num <= expected_num + GAP_TOLERANCE)

        # Decision: is this a real hadith boundary?
        accept_as_hadith = False

        if current_hadith is None:
            # No current hadith — accept if it has a narration keyword
            if has_keyword:
                accept_as_hadith = True
        else:
            if is_sequential:
                # Number follows the sequence — strong signal
                accept_as_hadith = True
            elif has_keyword and num >= current_hadith["num"]:
                # Has a keyword AND number is >= current — likely a real
                # hadith that we just had a gap for
                accept_as_hadith = True
            # else: this is commentary/bullet point — merge into previous

        if accept_as_hadith:
            skipped_count = 0  # Reset skip counter
            # Save previous hadith if it exists
            if current_hadith is not None:
                finalized = _finalize_chunk(current_hadith)
                if finalized is not None:
                    chunks.append(finalized)

            current_hadith = {
                "num": num,
                "num_str": cand["num_str"],
                "content": content
            }
            expected_num = num + 1
        else:
            skipped_count += 1

            # If we've skipped too many in a row, the state machine is
            # stuck on a bad sequence (e.g. preface numbered items).
            # Reset so we can latch onto the real hadith sequence.
            if skipped_count >= RESET_THRESHOLD and current_hadith is not None:
                finalized = _finalize_chunk(current_hadith)
                if finalized is not None:
                    chunks.append(finalized)
                current_hadith = None
                expected_num = None
                skipped_count = 0
                # Re-evaluate this candidate as a potential new start
                if has_keyword:
                    current_hadith = {
                        "num": num,
                        "num_str": cand["num_str"],
                        "content": content
                    }
                    expected_num = num + 1
            elif current_hadith is not None:
                # Merge into previous hadith's content
                current_hadith["content"] += f"\n{cand['num_str']}. {content}"
            # else: pre-hadith noise (intro, TOC), discard

    # Don't forget the last hadith
    if current_hadith is not None:
        finalized = _finalize_chunk(current_hadith)
        if finalized is not None:
            chunks.append(finalized)

    return chunks


def _finalize_chunk(hadith: Dict) -> Dict:
    """
    Clean up a hadith chunk's content and build the output dict.
    Preserves paragraph structure (double newlines) while cleaning up
    single newlines that are just PDF line-wrapping artifacts.
    """
    content = hadith["content"].strip()

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Preserve double newlines (paragraph breaks) by temporarily replacing them
    content = re.sub(r"\n\s*\n", "\n\n", content)  # normalize varied double newlines
    paragraphs = content.split("\n\n")

    # Within each paragraph, join single-newline-wrapped lines with spaces
    cleaned_paragraphs = []
    for para in paragraphs:
        # Replace single newlines with spaces (PDF line wrapping)
        joined = re.sub(r"\s*\n\s*", " ", para.strip())
        # Collapse multiple spaces
        joined = re.sub(r"  +", " ", joined)
        if joined:
            cleaned_paragraphs.append(joined)

    content = "\n\n".join(cleaned_paragraphs)

    # Skip chunks that are too short to be real hadiths
    if len(content) < 20:
        return None

    num_str = hadith["num_str"]
    return {
        "hadith_number": num_str,
        "content": f"Hadith {num_str}: {content}"
    }
