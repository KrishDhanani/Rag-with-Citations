import re

import chromadb
from rank_bm25 import BM25Okapi

from src import config

# small stopword list, enough to stop "the", "of" ... from dominating a match
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "his", "her",
    "their", "him", "them", "i", "you", "we", "what", "who", "whom", "which",
    "when", "where", "why", "how", "do", "does", "did", "had", "has", "have",
    "not", "no", "so", "if", "then", "there",
}


def tokenize(text):
    """Lowercase, split on non-alphanumeric characters, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def load_chunks_from_chroma():
    """Read all chunks back from the vector store so BM25 and dense search
    work on exactly the same corpus."""
    client = chromadb.PersistentClient(path=str(config.VECTOR_DB_DIR))
    collection = client.get_collection(config.COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])

    chunks, seen = [], set()
    for text, meta in zip(data["documents"], data["metadatas"]):
        if meta["chunk_id"] in seen:      # skip duplicated entries
            continue
        seen.add(meta["chunk_id"])
        chunks.append({"chunk_id": meta["chunk_id"], "text": text, "metadata": meta})
    return chunks


class SparseSearch:
    """BM25 keyword search. The index is small, so it is built in memory on startup."""

    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else load_chunks_from_chroma()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search(self, query, top_k=config.SPARSE_TOP_K):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:      # no query term matched at all
                break
            results.append({**self.chunks[i], "score": float(scores[i])})
        return results
