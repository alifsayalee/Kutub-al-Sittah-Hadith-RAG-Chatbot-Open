import fitz
import os
import re

def load_pdf_text(pdf_path: str, start_page: int = 0, end_page: int = None) -> str:
    """
    Extract text from a PDF file.
    Optionally restrict to a specific page range to skip intro/TOC.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    if end_page is None:
        end_page = len(doc)
    
    end_page = min(end_page, len(doc))
    
    text_blocks = []
    

    for page_num in range(start_page, end_page):
        page_text = doc[page_num].get_text("text")
        
        # Clean up obvious junk lines
        lines = page_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip empty lines
            if not line_stripped:
                continue
            # Skip URL footers
            if "wordpress.com" in line_stripped.lower() or "http" in line_stripped.lower():
                continue
            
            cleaned_lines.append(line_stripped)
            
        text_blocks.append("\n".join(cleaned_lines))

    doc.close()
    return "\n".join(text_blocks)
