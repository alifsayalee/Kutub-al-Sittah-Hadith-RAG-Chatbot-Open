"""
retrieval/retriever.py

Handles query embedding via Hugging Face Inference API and
ChromaDB vector search. (Phase 3a)
"""
import os
import chromadb
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "kutub_al_sittah")
TOP_K = int(os.getenv("TOP_K", "5"))

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=chromadb.config.Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

if HF_API_TOKEN:
    hf_client = InferenceClient(model="BAAI/bge-m3", token=HF_API_TOKEN)

def get_query_embedding(query_text: str) -> list[float]:
    """Embeds the query using Hugging Face Free API via InferenceClient."""
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN is missing from .env")
        
    vector = hf_client.feature_extraction(query_text)
    return vector.tolist()

def search_database(query_vector: list[float], book_filter: str = None) -> list[dict]:
    """Searches ChromaDB for the closest vectors."""
    where_clause = {"book": book_filter} if book_filter else None
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved = []
    if results["ids"] and len(results["ids"]) > 0:
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(ids)
        
        for i in range(len(ids)):
            retrieved.append({
                "id": ids[i],
                "text": docs[i],
                "book": metas[i].get("book_display", "Unknown Book"),
                "volume": metas[i].get("volume", "Unknown"),
                "hadith_number": metas[i].get("hadith_number", "Unknown"),
                "distance": distances[i]
            })
            
    return retrieved
