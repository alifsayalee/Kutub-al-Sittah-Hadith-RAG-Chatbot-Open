"""
retrieval/generator.py

Handles answering queries using Google Gemini API based on retrieved context.
"""
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

def generate_answer(query_text: str, retrieved_hadiths: list[dict]) -> str:
    """Uses Gemini to answer the question based only on the context."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing from .env")
        
    if not retrieved_hadiths:
        return "I could not find any relevant data to this, please consult a scholar."
        
    # Construct context block
    context_blocks = []
    for i, h in enumerate(retrieved_hadiths):
        context_blocks.append(
            f"[Source {i+1}] {h['book']}, Volume {h['volume']}, Hadith {h['hadith_number']}:\n{h['text']}"
        )
    context_str = "\n\n".join(context_blocks)
    
    prompt = f"""You are an expert Islamic RAG assistant. 
You must answer the user's question using ONLY the provided hadith context below.
Provide your answer in exactly 2-3 coherent paragraphs in easy to understand English.
If the context does not contain the answer, explicitly state that "I could not find any relevant data to this, please consult a scholar."
Do not hallucinate or bring in outside knowledge. 

CONTEXT:
{context_str}

USER QUESTION:
{query_text}

ANSWER:"""

    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    return response.text
