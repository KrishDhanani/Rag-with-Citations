import json
from unittest.mock import Mock

import pytest
import requests
from langchain_core.documents import Document

from src.ingestion.chunker import Chunking
from src.ingestion.contextualizer import Contextualizer, search_text, embedding_text
from src.ingestion.parser import Parser
from src.ingestion.vector_store import Vector_Database


def doc(text, page=1):
    return Document(page_content=text, metadata={"source": "book.pdf", "page": page,
                                                "section": "Chapter 1", "file_type": "pdf"})


class WordTokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split() + (["<start>", "<end>"] if add_special_tokens else [])

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)

    def num_special_tokens_to_add(self, pair=False):
        return 2


def test_context_cannot_truncate_original_passage():
    text = "Original evidence must remain complete even with a long contextual description."
    metadata = {"contextual_description": "Harry visits Ollivander. " * 100}
    enriched = embedding_text(text, metadata, WordTokenizer(), 24)
    assert enriched.endswith(text)
    assert len(WordTokenizer().encode(enriched)) <= 24


def test_paragraphs_preserve_dialogue_and_overlap():
    paragraphs = ["Harry asked a question.", '"A wand chooses its wizard," said Ollivander.',
                  "Harry picked up another wand."]
    chunks = Chunking(80, 45).chunk_documents([doc("\n\n".join(paragraphs))])
    assert chunks[0].page_content == "\n\n".join(paragraphs[:2])
    assert chunks[1].page_content == "\n\n".join(paragraphs[1:])
    assert all(len(c.page_content) <= 80 for c in chunks)


def test_oversized_paragraph_and_page_metadata():
    chunks = Chunking(80, 10).chunk_documents([doc("word " * 100), doc("Next page.", 2)])
    assert all(len(c.page_content) <= 80 for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    assert chunks[-1].metadata["page"] == 2
    assert chunks[-1].metadata["section"] == "Chapter 1"


def test_pdf_blocks_preserve_paragraphs_and_pdf_page_numbers(tmp_path):
    import pymupdf
    path = tmp_path / "book.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 60), "First paragraph wraps\nonto another line.")
        page.insert_text((50, 140), "Second paragraph.")
        page.insert_text((50, 750), "99")
        pdf.save(path)
    documents = Parser().parse_document(path)
    assert "\n\nSecond paragraph." in documents[0].page_content
    assert "99" not in documents[0].page_content
    assert documents[0].metadata["page"] == 1


def test_context_cached_per_page_and_original_evidence_unchanged(tmp_path, monkeypatch):
    response = Mock()
    response.json.return_value = {"done": True, "done_reason": "stop", "message": {
        "content": json.dumps({"context": ["Harry", "Ollivander", "tries a wand"]})}}
    post = Mock(return_value=response)
    monkeypatch.setattr("src.ingestion.contextualizer.requests.post", post)
    documents = [doc("Harry visits Ollivander.\n\nHe tries a wand.")]
    chunks = Chunking(30, 0).chunk_documents(documents)
    original = [c.page_content for c in chunks]
    contextualizer = Contextualizer(cache_dir=tmp_path)
    contextualizer.enrich(documents, chunks)
    contextualizer.enrich(documents, chunks)
    assert post.call_count == 1
    assert [c.page_content for c in chunks] == original
    assert "Ollivander" in search_text(chunks[0].page_content, chunks[0].metadata)
    contextualizer.model = "different-model"
    contextualizer.enrich(documents, chunks)
    assert post.call_count == 2


def test_bad_model_output_uses_cached_source_excerpt(tmp_path, monkeypatch):
    response = Mock()
    response.json.return_value = {"done": True, "done_reason": "stop", "message": {"content": "{}"}}
    monkeypatch.setattr("src.ingestion.contextualizer.requests.post", Mock(return_value=response))
    chunks = Chunking().chunk_documents([doc("Evidence")])
    Contextualizer(cache_dir=tmp_path).enrich([doc("Evidence")], chunks)
    assert chunks[0].metadata["contextual_description"] == "Evidence"
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["context"] == []


def test_invented_context_claims_are_discarded():
    result = Contextualizer._validate({"context": ["Harry", "Harry will be in Hufflepuff"]},
                                      {"target": "Harry visits a wand shop.", "surrounding_pages": []})
    assert result == "Harry"


def test_all_invented_context_falls_back_to_original_source():
    result = Contextualizer._validate({"context": ["Harry will be in Hufflepuff"]},
                                      {"target": "Harry visits a wand shop.", "surrounding_pages": []})
    assert result == "Harry visits a wand shop."


def test_ollama_token_repeat_abort_uses_source_excerpt(tmp_path, monkeypatch):
    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error":"prediction aborted, token repeat limit reached"}'
    monkeypatch.setattr("src.ingestion.contextualizer.requests.post", Mock(return_value=response))
    chunks = Chunking().chunk_documents([doc("Original source.")])
    Contextualizer(cache_dir=tmp_path).enrich([doc("Original source.")], chunks)
    assert chunks[0].metadata["contextual_description"] == "Original source."


def test_connection_failure_retries_without_caching(tmp_path, monkeypatch):
    post = Mock(side_effect=requests.ConnectionError("Ollama unavailable"))
    monkeypatch.setattr("src.ingestion.contextualizer.requests.post", post)
    monkeypatch.setattr("src.ingestion.contextualizer.time.sleep", Mock())
    with pytest.raises(requests.ConnectionError):
        Contextualizer(cache_dir=tmp_path).enrich([doc("Evidence")], Chunking().chunk_documents([doc("Evidence")]))
    assert post.call_count == 3
    assert not list(tmp_path.glob("*.json"))


def test_storage_embeds_context_but_stores_original_and_removes_stale(monkeypatch):
    store = Mock()
    store.get.return_value = {"ids": ["obsolete"]}
    monkeypatch.setattr("src.ingestion.vector_store.Chroma", Mock(return_value=store))
    database = Vector_Database.__new__(Vector_Database)
    database.DIR = "unused"
    database.embeddings = Mock()
    database.embeddings._client.tokenizer = WordTokenizer()
    database.embeddings._client.max_seq_length = 256
    database.embeddings.embed_documents.return_value = [[0.1, 0.2]]
    chunk = Chunking().chunk_documents([doc("He tried another wand.")])[0]
    chunk.metadata["contextual_description"] = "Harry visits Ollivander."
    database.store_data([chunk])
    assert "Harry visits Ollivander" in database.embeddings.embed_documents.call_args.args[0][0]
    assert store._collection.upsert.call_args.kwargs["documents"] == ["He tried another wand."]
    store.delete.assert_called_once_with(ids=["obsolete"])
