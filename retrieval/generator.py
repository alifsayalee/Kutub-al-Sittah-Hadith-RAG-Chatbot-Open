"""
retrieval/generator.py

Handles answering queries using the OpenRouter API (grounded in retrieved context).
Uses requests for maximum portability without external heavy SDK dependencies.
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

def generate_answer(query_text: str, retrieved_hadiths: list[dict]) -> str:
    """Uses OpenRouter to answer the question based only on the retrieved context."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing from .env")
        
    if not retrieved_hadiths:
        return "I could not find any relevant data to this, please consult a scholar."
        
    # Construct context block
    context_blocks = []
    for i, h in enumerate(retrieved_hadiths):
        context_blocks.append(
            f"[Source {i+1}] {h['book']}, Volume {h['volume']}, Hadith {h['hadith_number']}:\n{h['text']}"
        )
    context_str = "\n\n".join(context_blocks)
    
    # Highly refined, respectful, family-friendly system prompt
    system_prompt = """You are an expert Islamic RAG assistant. 
You must answer the user's question using ONLY the provided hadith context.
Provide your answer in exactly 2-3 coherent paragraphs in easy to understand, respectful, and elegant English.

CRITICAL INSTRUCTIONS:
1. Focus on the spiritual, moral, social, and practical teachings of the Prophet (peace be upon him) regarding the query.
2. If the retrieved context contains dry linguistic comments or graphic dictionary definitions of Arabic root words (such as literal references to intercourse under the definition of Nikah), you MUST bypass them entirely. Focus instead on the metaphorical, moral, and sacred bond of the marriage covenant.
3. Keep the tone highly professional, respectful, family-friendly, and educational.

If the context does not contain the answer, or if the user's question is gibberish, invalid, not in English, or completely out-of-scope relative to the context, you MUST output exactly: "[REFUSAL_SHIELD]" and absolutely nothing else. Do not explain, apologize, or write any other words.
Do not bring in outside knowledge."""

    user_prompt = f"CONTEXT:\n{context_str}\n\nUSER QUESTION:\n{query_text}\n\nANSWER:"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        result_json = response.json()
        
        if "choices" in result_json and len(result_json["choices"]) > 0:
            return result_json["choices"][0]["message"]["content"]
        else:
            return f"Error: Unexpected response format from OpenRouter API: {result_json}"
            
    except Exception as e:
        return f"An LLM generation error occurred: {e}"
