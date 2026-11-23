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

def test_engine_includes_pdf_page_context():
    trial = make_chunk("trial", "Try the beech wand.", page=67)
    chosen = make_chunk("chosen", "The chosen wand was holly and phoenix feather.", page=67)
    for index, chunk in enumerate((trial, chosen), 1):
        chunk["metadata"].update(file_type="pdf", chunk_index=index)
    dense = FakeDense()
    dense.search = lambda query, top_k: [{**trial, "score": 0.9}]
    engine = RetrievalEngine(
        dense=dense,
        sparse=SparseSearch([trial, chosen]),
        reranker=Reranker(model=FakeCrossEncoder()),
    )
    result = engine.retrieve("beech", top_k=1)
    assert "chosen wand was holly" in result["context"]
    assert result["chunks"][0]["page"] == 67

def test_parent_identity_follows_names_in_evidence():
    from unittest.mock import Mock

    initial = make_chunk("parents", "Nora's parents, Mira and Theo Vale, are dead.", page=10)
    identifying = make_chunk("mother", "Nora is Mira Vale's daughter.", page=42)
    for chunk in (initial, identifying):
        chunk["metadata"].update(file_type="pdf", chunk_index=1)
    dense = Mock()
    dense.search.side_effect = [[initial], [initial], [identifying]]
    sparse = Mock()
    sparse.chunks = [initial, identifying]
    sparse.search.side_effect = [[], [], [identifying]]
    engine = RetrievalEngine(
        dense=dense, sparse=sparse, reranker=Reranker(model=FakeCrossEncoder()),
    )
    result = engine.retrieve("Who is Noras mom ?")

    assert result["query"] == "Who is Noras mom ?"
    assert "Nora is Mira Vale's daughter." in result["context"]
    assert "Mira" in sparse.search.call_args_list[-1].args[0]
    assert "Vale" in sparse.search.call_args_list[-1].args[0]
    assert "Nora's parents, Mira and Theo Vale, are dead." in result["context"]
    assert result["top_score"] == result["chunks"][0]["confidence"]


def test_parent_query_preserves_original_relationship_and_adds_variant():
    from unittest.mock import Mock
    dense = Mock()
    dense.search.return_value = [TOY_CHUNKS[0]]
    sparse = SparseSearch(TOY_CHUNKS)
    sparse.search = Mock(return_value=[])
    reranker = Mock()
    reranker.rerank.return_value = [{**TOY_CHUNKS[0], "confidence": 0.75}]
    result = RetrievalEngine(dense=dense, sparse=sparse, reranker=reranker).retrieve(
        "Who is Nora's mom?")
    assert [call.args[0] for call in dense.search.call_args_list] == [
        "Who is Nora's mother?", "Who are Nora's parents?"]
    assert reranker.rerank.call_args.args[0] == "Who is Nora's mother?"
    assert result["query"] == "Who is Nora's mom?"


def test_followup_uses_final_reranker_confidence():
    from unittest.mock import Mock
    initial = make_chunk("parents", "Nora's parents, Mira and Theo Vale, are dead.", page=10)
    linked = make_chunk("mother", "Nora is Mira Vale's daughter.", page=42)
    for chunk in (initial, linked):
        chunk["metadata"].update(file_type="pdf", chunk_index=1)
    dense = Mock()
    dense.search.side_effect = [[initial], [initial], [linked]]
    sparse = Mock()
    sparse.chunks = [initial, linked]
    sparse.search.side_effect = [[], [], [linked]]
    reranker = Mock()
    reranker.rerank.side_effect = [
        [{**initial, "confidence": 0.95}],
        [{**linked, "confidence": 0.25}, {**initial, "confidence": 0.2}],
    ]
    result = RetrievalEngine(dense=dense, sparse=sparse, reranker=reranker).retrieve(
        "Who is Nora's mother?")
    assert result["top_score"] == 0.25
    assert {item["chunk_id"] for item in reranker.rerank.call_args.args[1]} == {"parents", "mother"}


def test_sibling_pairs_do_not_supply_parent_name_hints():
    siblings = make_chunk("siblings", "Nora met Fred and George Vale and their mother.")
    parents = make_chunk("parents", "Nora was Mira and Theo Vale' s daughter.", page=2)
    for item in (siblings, parents):
        item["metadata"].update(file_type="pdf", chunk_index=1)
    engine = RetrievalEngine(dense=FakeDense(), sparse=SparseSearch([siblings, parents]),
                             reranker=Reranker(model=FakeCrossEncoder()))
    assert engine._family_name_hints("Nora") == ["Mira", "Theo", "Vale"]
    assert engine._family_name_hints("Other") == []


def test_paragraph_candidates_keep_complete_page_dialogue_without_overlap():
    from unittest.mock import Mock
    paragraphs = [make_chunk(f"p{i}", f"Paragraph {i}.", page=67) for i in range(6)]
    paragraphs[3]["text"] = "Paragraph 2.\n\nParagraph 3."
    for index, chunk in enumerate(paragraphs):
        chunk["metadata"].update(file_type="pdf", chunk_index=index,
                                 chunking_version="paragraph-v1")
    dense = Mock()
    dense.search.return_value = [paragraphs[2]]
    sparse = SparseSearch(paragraphs)
    sparse.search = Mock(return_value=[])
    engine = RetrievalEngine(dense=dense, sparse=sparse,
                             reranker=Reranker(model=FakeCrossEncoder()))
    result = engine.retrieve("Paragraph 2", top_k=1)
    assert "Paragraph 1." in result["context"]
    assert "Paragraph 3." in result["context"]
    assert "Paragraph 0." in result["context"]
    assert "Paragraph 5." in result["context"]
    assert result["context"].count("Paragraph 2.") == 1


def test_bm25_searches_generated_context_but_returns_original_evidence():
    chunks = [make_chunk("wand", "He tried another one."),
              make_chunk("dog", "A dog barked."), make_chunk("train", "A train left.")]
    chunks[0]["metadata"]["contextual_description"] = "Harry visits Ollivanders."
    result = SparseSearch(chunks).search("Ollivanders")
    assert result[0]["text"] == "He tried another one."
    assert "Ollivanders" not in build_context(tag_citations(result))
