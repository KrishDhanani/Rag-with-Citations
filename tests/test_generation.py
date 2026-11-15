from unittest.mock import Mock

import pytest

from src.generation.generator import GenerationEngine
from src.generation.refusal import should_refuse


@pytest.mark.parametrize(
    ("context", "score", "expected"),
    [
        ("[1] Nicolas Flamel is a noted alchemist.", 0.997, False),
        ("[1] Harry said he was boiling.", 0.000017, True),
        ("   ", 0.997, True),
        ("[1] Retrieved without reranking.", None, True),
        ("[1] Evidence at the threshold.", 0.5, False),
        ("[1] Invalid score.", float("nan"), True),
    ],
)
def test_should_refuse(context, score, expected):
    assert should_refuse(context, score, threshold=0.5) is expected


@pytest.fixture
def retrieval_result():
    return {
        "query": "Who is Nicolas Flamel?",
        "context": "[1] Flamel was an alchemist.\n[2] Harry was curious.",
        "top_score": 0.997,
        "chunks": [
            {"citation_id": "[1]", "document": "book.pdf", "page": 177,
             "section": None, "snippet": "Flamel was an alchemist."},
            {"citation_id": "[2]", "document": "book.pdf", "page": 158,
             "section": None, "snippet": "Harry was curious."},
        ],
    }


def test_generator_returns_only_cited_sources(retrieval_result):
    client = Mock()
    answer = "Flamel was an alchemist [1]."
    client.complete.return_value = '{"status": "answered", "answer": "Flamel was an alchemist [1]."}'
    result = GenerationEngine(client, threshold=0.5).generate(retrieval_result)

    assert result == {
        "status": "answered",
        "answer": answer,
        "citations": [{"id": "[1]", "document": "book.pdf", "page": 177,
                       "section": None, "snippet": "Flamel was an alchemist."}],
    }
    client.complete.assert_called_once_with(retrieval_result["query"], retrieval_result["context"])


def test_generator_skips_model_for_weak_retrieval(retrieval_result):
    client = Mock()
    retrieval_result["top_score"] = 0.000017
    result = GenerationEngine(client, threshold=0.5).generate(retrieval_result)

    assert result["status"] == "refused"
    assert result["citations"] == []
    client.complete.assert_not_called()


@pytest.mark.parametrize("response", [
    '{"status": "refused", "answer": ""}',
    '{"status": "answered", "answer": "An alchemist [1] [9]."}',
    '{"status": "answered", "answer": "An alchemist."}',
    "invalid JSON",
])
def test_generator_refuses_unusable_model_output(retrieval_result, response):
    client = Mock()
    client.complete.return_value = response
    result = GenerationEngine(client, threshold=0.5).generate(retrieval_result)

    assert result["status"] == "refused"
    assert result["citations"] == []
