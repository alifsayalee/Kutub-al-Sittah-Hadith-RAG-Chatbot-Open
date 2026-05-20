"""
ingestion/embedder.py

Local embedding using BAAI/bge-m3 via sentence-transformers.
No API key required — the model runs entirely on your machine (CPU).

Model: BAAI/bge-m3
- Output dimension: 1024
- Downloads once to ~/.cache/huggingface (~2.2 GB)
- Optimised for both document and query embedding

RAM-safe settings for an 8 GB system:
- batch_size=12: processes 12 texts at a time
- Single-threaded: no extra workers spawned
"""

import os
import sys
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Limit PyTorch to 6 threads to balance speed vs CPU contention
torch.set_num_threads(6)

load_dotenv()


class HadithEmbedder:
    """
    Wraps BAAI/bge-m3 with CPU-safe batching.
    The model is loaded once at construction time and reused for all calls.
    """

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self, show_progress: bool = True):
        print(f"[EMBEDDER] Loading {self.MODEL_NAME} on CPU...")
        print(f"           (First run downloads ~2.2 GB to HuggingFace cache)")
        sys.stdout.flush()

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device="cpu",
        )
        self.show_progress = show_progress
        # Reduced from 12 to 4 to prevent RAM/SSD thrashing
        self.batch_size = 6
        self.model_name = self.MODEL_NAME

        dim = self.model.get_embedding_dimension()
        print(f"[EMBEDDER] Model ready. Output dimension: {dim}")
        sys.stdout.flush()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document strings for ingestion.
        Returns a list of plain Python float lists.
        Uses prompt_name='passage' for document encoding (bge-m3 best practice).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            prompt_name="document",     # bge-m3 uses 'document' for indexing
            normalize_embeddings=True,  # cosine similarity works best normalised
            convert_to_numpy=True,
        )
        # Convert numpy arrays to plain Python lists for ChromaDB compatibility
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query string for retrieval.
        Uses prompt_name='query' for query encoding (bge-m3 best practice).
        Returns a single plain Python float list.
        """
        vec = self.model.encode(
            query,
            prompt_name="query",        # bge-m3 best practice for queries
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()
