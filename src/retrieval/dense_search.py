import chromadb
from sentence_transformers import SentenceTransformer

from src import config


class DenseSearch:
    """Semantic search over the Chroma collection built in the ingestion step."""

    def __init__(self, collection=None, model=None):
        if collection is None:
            client = chromadb.PersistentClient(path=str(config.VECTOR_DB_DIR))
            collection = client.get_collection(config.COLLECTION_NAME)
        self.collection = collection
        self.model = model or SentenceTransformer(config.EMBEDDING_MODEL)

    def search(self, query, top_k=config.DENSE_TOP_K):
        query_vec = self.model.encode(query, normalize_embeddings=True).tolist()

        # The collection currently holds every chunk twice (ingestion was run
        # two times), so we ask for more results and drop repeated chunk_ids.
        n = min(top_k * 3, self.collection.count())
        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        results, seen = [], set()
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk_id = meta["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            # Chroma returns squared L2 distance. The embeddings are unit
            # length, so cosine similarity = 1 - d^2 / 2.
            results.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta,
                "score": 1 - dist / 2,
            })
            if len(results) == top_k:
                break
        return results
