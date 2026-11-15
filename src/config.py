import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# ---- Retrieval settings (Dev 2) ----
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "Vectorstore"
COLLECTION_NAME = "rag_collection"

# must be the same model that was used during ingestion
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_TOP_K = 20      # candidates taken from vector search
SPARSE_TOP_K = 20     # candidates taken from BM25
RRF_K = 60            # constant from the RRF paper (Cormack et al., 2009)
RERANK_TOP_N = 20     # how many fused candidates go to the cross-encoder
FINAL_TOP_K = 5       # chunks handed to the generator
