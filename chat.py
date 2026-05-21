import sys
import os
import re
from dotenv import load_dotenv

# Ensure virtual env or package installation is detected
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.text import Text

from retrieval.retriever import get_query_embedding, search_database
from retrieval.generator import generate_answer
from retrieval.api_fetcher import fetch_clean_hadith

# Initialize Rich Console
console = Console()

# Global state to track sources in the current session
last_retrieved_sources = []

BOOK_MAPPING = {
    "0": None,
    "1": "bukhari",
    "2": "muslim",
    "3": "abu_dawud",
    "4": "tirmidhi",
    "5": "nasai",
    "6": "ibn_majah"
}

BOOK_DISPLAY_NAMES = {
    None: "All Kutub al-Sittah Books",
    "bukhari": "Sahih al-Bukhari",
    "muslim": "Sahih Muslim",
    "abu_dawud": "Sunan Abu Dawud",
    "tirmidhi": "Jami` at-Tirmidhi",
    "nasai": "Sunan an-Nasai",
    "ibn_majah": "Sunan Ibn Majah"
}

def display_welcome_banner():
    """Prints a beautiful welcome panel to start the application."""
    console.print(Panel(
        "[bold green]🕌 Kutub al-Sittah Hadith RAG Assistant 🕌[/bold green]\n"
        "[italic white]An intelligent semantic search & generation system grounded in the six canonical Hadith collections.[/italic white]",
        border_style="green",
        expand=False,
        padding=(1, 2)
    ))

def select_book_filter() -> str:
    """Prompts the user to select which book to search."""
    console.print("\n[bold cyan]Select target Hadith collection for your queries:[/bold cyan]")
    console.print(" [0] Search All Books (Default)")
    console.print(" [1] Sahih al-Bukhari")
    console.print(" [2] Sahih Muslim")
    console.print(" [3] Sunan Abu Dawud")
    console.print(" [4] Jami` at-Tirmidhi")
    console.print(" [5] Sunan an-Nasai")
    console.print(" [6] Sunan Ibn Majah")
    
    choice = Prompt.ask("[bold yellow]Enter choice (0-6)[/bold yellow]", choices=[str(i) for i in range(7)], default="0")
    selected_book = BOOK_MAPPING[choice]
    
    console.print(f"[bold green]✓ Searching set to: [yellow]{BOOK_DISPLAY_NAMES[selected_book]}[/yellow][/bold green]\n")
    return selected_book

def sanitize_hadith_text(raw_text: str) -> str:
    """Light text cleanup preserving all Unicode characters (Arabic, salawat symbols, etc.)."""
    if not raw_text:
        return ""
    # Normalise whitespace only — the clean dataset needs no OCR sanitisation
    text = re.sub(r'\s+', ' ', raw_text).strip()
    return text

def get_word_overlap_similarity(text1: str, text2: str) -> float:
    """Computes Jaccard similarity of normalized words to detect mismatched hadiths across editions."""
    # Lowercase and keep only alphabetical words of length >= 3
    words1 = set(re.findall(r'\b[a-z]{3,}\b', text1.lower()))
    words2 = set(re.findall(r'\b[a-z]{3,}\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
        
    # Filter standard high-frequency stop words to focus on content matching
    stopwords = {
        "the", "and", "that", "was", "for", "with", "his", "him", "her", 
        "they", "them", "this", "had", "not", "but", "she", "you", "are", 
        "our", "out", "from", "has", "have", "been", "were", "who", "which"
    }
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

def handle_view_command(clean_query: str):
    """Processes `/view <number>` commands statefully."""
    global last_retrieved_sources
    if not last_retrieved_sources:
        console.print("[bold red]⚠ No references are currently active. Ask a question first![/bold red]\n")
        return
        
    try:
        parts = clean_query.split()
        if len(parts) < 2:
            console.print("[bold red]⚠ Format is: /view <number> (e.g., /view 1)[/bold red]\n")
            return
            
        index = int(parts[1]) - 1
        if index < 0 or index >= len(last_retrieved_sources):
            console.print(f"[bold red]⚠ Invalid index. Choose a number between 1 and {len(last_retrieved_sources)}[/bold red]\n")
            return
            
        source = last_retrieved_sources[index]
        
        # Map display name to raw book key used by Hadith API
        DISPLAY_TO_RAW_MAPPING = {
            "Sahih al-Bukhari": "bukhari",
            "Sahih Muslim": "muslim",
            "Sunan Abu Dawud": "abu_dawud",
            "Jami` at-Tirmidhi": "tirmidhi",
            "Jami at-Tirmidhi": "tirmidhi",
            "Sunan an-Nasai": "nasai",
            "Sunan Ibn Majah": "ibn_majah"
        }
        
        display_book = source["book"]
        raw_book_key = DISPLAY_TO_RAW_MAPPING.get(display_book, "")
        hadith_num = source["hadith_number"]
        
        cleaned_local_text = sanitize_hadith_text(source["text"])
        cleaned_text = None
        is_api_sourced = False
        similarity_score = 0.0
        
        # Try fetching from digital API first
        if raw_book_key:
            with console.status("[bold green]Fetching pristine translation from API...", spinner="dots"):
                api_text = fetch_clean_hadith(raw_book_key, hadith_num)
                if api_text:
                    similarity_score = get_word_overlap_similarity(cleaned_local_text, api_text)
                    # If similarity is >= 0.15, we accept the online text as matching
                    if similarity_score >= 0.15:
                        cleaned_text = api_text
                        is_api_sourced = True
                    else:
                        console.print(f"[dim yellow]ℹ Note: Online database numbering mismatch detected (Similarity: {similarity_score:.2f}). Fallback to local text.[/dim yellow]")
        
        # Local database failsafe if offline, mismatched, or rate-limited
        if not cleaned_text:
            cleaned_text = cleaned_local_text
            
        meta_title = f"📖 {source['book']}, Vol {source['volume']}, Hadith {source['hadith_number']}"
        if is_api_sourced:
            meta_label = f"[bold yellow]{meta_title}[/bold yellow] [bold green](Pristine Digital)[/bold green]"
        else:
            meta_label = f"[bold yellow]{meta_title}[/bold yellow] [bold cyan](Local Failsafe)[/bold cyan]"
            
        console.print(Panel(
            cleaned_text,
            title=meta_label,
            title_align="left",
            border_style="blue",
            padding=(1, 2)
        ))
        console.print("[dim]Type another question, or type another '/view <number>' command.[/dim]\n")
        
    except ValueError:
        console.print("[bold red]⚠ Please specify a valid integer index (e.g., /view 1)[/bold red]\n")


def start_chat_loop():
    global last_retrieved_sources
    display_welcome_banner()
    current_filter = select_book_filter()
    
    console.print("[dim]Type your question and press Enter. Commands available:[/dim]")
    console.print("[dim] - '/filter' to change active book[/dim]")
    console.print("[dim] - '/clear' to clear screen[/dim]")
    console.print("[dim] - '/view <number>' to read full source details[/dim]")
    console.print("[dim] - 'exit' or 'quit' to close app[/dim]\n")
    
    while True:
        try:
            query = Prompt.ask("[bold magenta]Question[/bold magenta]")
            clean_query = query.strip()
            if not clean_query:
                continue
                
            if clean_query.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold green]Peace be upon you! Exiting...[/bold green]")
                break
                
            if clean_query == "/clear":
                console.clear()
                display_welcome_banner()
                console.print(f"[bold green]Active Search: [yellow]{BOOK_DISPLAY_NAMES[current_filter]}[/yellow][/bold green]\n")
                continue
                
            if clean_query == "/filter":
                current_filter = select_book_filter()
                last_retrieved_sources = [] # Reset state
                continue
                
            if clean_query.startswith("/view"):
                handle_view_command(clean_query)
                continue
            
            # Execute standard RAG query
            with console.status("[bold green]Searching database & generating answer...", spinner="dots"):
                # Embed query
                query_vector = get_query_embedding(clean_query)
                
                # Fetch closest matches from ChromaDB
                retrieved_hadiths = search_database(query_vector, book_filter=current_filter)
                
                # Check for empty results
                if not retrieved_hadiths:
                    answer = "I could not find any relevant data to this, please consult a scholar."
                    is_fallback = True
                else:
                    # Layer 1: Distance-Based Retriever Shield
                    # Cosine distance > 0.45 denotes highly unrelated vectors
                    top_distance = retrieved_hadiths[0].get("distance", 1.0)
                    if top_distance > 0.42:
                        answer = "I could not find any relevant data to this, please consult a scholar."
                        is_fallback = True
                    else:
                        answer = generate_answer(clean_query, retrieved_hadiths)
                        # Layer 2: LLM Sentinel Refusal Guard
                        if "[REFUSAL_SHIELD]" in answer:
                            answer = "I could not find any relevant data to this, please consult a scholar."
                            is_fallback = True
                        else:
                            is_fallback = False
            
            # Display LLM response
            console.print("\n" + "=" * 60)
            console.print(Panel(
                Markdown(answer) if not is_fallback else Text(answer),
                title="[bold green]🕌 Synthesized Answer[/bold green]" if not is_fallback else "[bold red]⚠ Notice[/bold red]",
                title_align="left",
                border_style="green" if not is_fallback else "red",
                padding=(1, 2)
            ))
            
            # Display metadata list ONLY if it is not a fallback
            if not is_fallback and retrieved_hadiths:
                last_retrieved_sources = retrieved_hadiths  # Update state
                
                console.print("\n[bold cyan]📖 Referenced Hadith Sources:[/bold cyan]")
                for idx, h in enumerate(retrieved_hadiths):
                    console.print(f" [bold yellow][{idx+1}][/bold yellow] {h['book']}, Vol {h['volume']}, Hadith {h['hadith_number']}")
                console.print("\n[dim]To read the full clean text of any reference, type '/view <number>' (e.g., '/view 1').[/dim]")
            else:
                last_retrieved_sources = []  # Clear state
                
            console.print("=" * 60 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[bold green]Peace be upon you! Exiting...[/bold green]")
            break
        except Exception as e:
            console.print(f"\n[bold red]An error occurred: {e}[/bold red]\n")

if __name__ == "__main__":
    start_chat_loop()
