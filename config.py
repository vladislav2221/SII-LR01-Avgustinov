"""Central configuration for the RAG document assistant."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / ".chroma"))).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "lab01_documents")
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# Chunks are 320 tokens, with 48-token overlap (15%).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "320"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "48"))
TOP_K = int(os.getenv("TOP_K", "2"))
# Chroma cosine distance is converted to similarity = 1 - distance.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
