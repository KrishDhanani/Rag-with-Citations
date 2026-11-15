"""
Tests for the retrieval part. The two neural models are replaced by small
fakes, so the tests run fast and without downloading anything.

Run from the repo root:  python -m pytest tests/test_retrieval.py
"""
import pytest

from src import config
from src.retrieval.dense_search import DenseSearch
from src.retrieval.sparse_search import SparseSearch, tokenize, load_chunks_from_chroma
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import RetrievalEngine, tag_citations, build_context


def make_chunk(chunk_id, text, page=1):
    meta = {"chunk_id": chunk_id, "source": "book.pdf", "page": page, "section": "General"}
    return {"chunk_id": chunk_id, "text": text, "metadata": meta}


TOY_CHUNKS = [
    make_chunk("c1", "Hagrid kept a three-headed dog called Fluffy.", page=1),
    make_chunk("c2", "Harry plays Seeker for the Gryffindor Quidditch team.", page=2),
    make_chunk("c3", "The Hogwarts Express leaves from platform nine and three-quarters.", page=3),
    make_chunk("c4", "Hermione spent the evening in the library reading.", page=4),
]


class FakeDense:
    """Returns the toy chunks in a fixed order, like a dense retriever would."""
    def search(self, query, top_k=20):
        return [{**c, "score": 0.5} for c in (TOY_CHUNKS[1], TOY_CHUNKS[0], TOY_CHUNKS[3])]


class FakeCrossEncoder:
    """Scores a pair by word overlap between query and passage."""
    def predict(self, pairs):
        return [len(set(tokenize(q)) & set(tokenize(p))) - 1.0 for q, p in pairs]


class FakeCollection:
    """Mimics the Chroma collection, including the duplicated entries."""
    def count(self):
        return 4

    def query(self, query_embeddings, n_results, include):
        hits = [TOY_CHUNKS[0], TOY_CHUNKS[0], TOY_CHUNKS[1], TOY_CHUNKS[1]]
        return {
            "documents": [[h["text"] for h in hits]],
            "metadatas": [[h["metadata"] for h in hits]],
            "distances": [[0.2, 0.2, 0.9, 0.9]],
        }


class FakeEmbedder:
    def encode(self, text, normalize_embeddings=True):
        import numpy as np
        return np.array([1.0, 0.0])


def test_dense_search_removes_duplicates_and_converts_distance():
    results = DenseSearch(collection=FakeCollection(), model=FakeEmbedder()).search("dog", top_k=2)

    assert [r["chunk_id"] for r in results] == ["c1", "c2"]
    assert results[0]["score"] == pytest.approx(0.9)     # 1 - 0.2 / 2


def test_tokenize_lowercases_and_removes_stopwords():
    assert tokenize("The Boy Who LIVED!") == ["boy", "lived"]


def test_bm25_finds_keyword_match():
    results = SparseSearch(TOY_CHUNKS).search("Fluffy the dog")
    assert results[0]["chunk_id"] == "c1"


def test_bm25_returns_nothing_without_overlap():
    assert SparseSearch(TOY_CHUNKS).search("dragon egg") == []


def test_rrf_prefers_chunk_found_by_both_retrievers():
    dense = [TOY_CHUNKS[0], TOY_CHUNKS[1]]
    sparse = [TOY_CHUNKS[2], TOY_CHUNKS[1]]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    assert fused[0]["chunk_id"] == "c2"      # rank 2 in both lists beats rank 1 in one
    assert fused[0]["rrf_score"] == pytest.approx(2 / 62)
    assert len(fused) == 3                   # c2 must not appear twice
    assert fused[0]["dense_rank"] == 2 and fused[0]["sparse_rank"] == 2


def test_reranker_orders_by_model_score():
    reranker = Reranker(model=FakeCrossEncoder())
    out = reranker.rerank("Which position does Harry play in Quidditch?", TOY_CHUNKS, top_k=2)

    assert out[0]["chunk_id"] == "c2"
    assert len(out) == 2
    assert 0 < out[0]["confidence"] < 1


def test_citation_tags_and_context():
    tagged = tag_citations(TOY_CHUNKS[:2])
    assert [c["citation_id"] for c in tagged] == ["[1]", "[2]"]
    assert tagged[0]["document"] == "book.pdf" and tagged[0]["page"] == 1

    context = build_context(tagged)
    assert "[1] Source: book.pdf, page 1" in context
    assert "Fluffy" in context and "[2]" in context


def test_engine_end_to_end_with_fakes():
    engine = RetrievalEngine(
        dense=FakeDense(),
        sparse=SparseSearch(TOY_CHUNKS),
        reranker=Reranker(model=FakeCrossEncoder()),
    )
    result = engine.retrieve("Which position does Harry play in Quidditch?", top_k=3)

    ids = [c["chunk_id"] for c in result["chunks"]]
    assert ids[0] == "c2"
    assert len(ids) == len(set(ids))
    assert result["top_score"] == result["chunks"][0]["confidence"]
    assert result["context"].startswith("[1]")


def test_engine_handles_empty_query():
    engine = RetrievalEngine(dense=FakeDense(), sparse=SparseSearch(TOY_CHUNKS),
                             reranker=Reranker(model=FakeCrossEncoder()))
    assert engine.retrieve("   ")["chunks"] == []


@pytest.mark.skipif(not config.VECTOR_DB_DIR.exists(), reason="vector store not built")
def test_bm25_on_real_corpus():
    chunks = load_chunks_from_chroma()
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))         # duplicates from double ingestion removed

    results = SparseSearch(chunks).search("Who is Nicolas Flamel?", top_k=5)
    assert any("Flamel" in r["text"] for r in results)
