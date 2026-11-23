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


def test_generator_normalizes_grouped_citations(retrieval_result):
    client = Mock()
    client.complete.return_value = '{"status": "answered", "answer": "Flamel was an alchemist [1, 2]."}'
    result = GenerationEngine(client, threshold=0.5).generate(retrieval_result)

    assert result["answer"] == "Flamel was an alchemist [1] [2]."
    assert [citation["id"] for citation in result["citations"]] == ["[1]", "[2]"]


@pytest.mark.parametrize("response", [
    '{"status": "refused", "answer": ""}',
    '{"status": "answered", "answer": "An alchemist [1] [9]."}',
    '{"status": "answered", "answer": "An alchemist [1, 9]."}',
    '{"status": "answered", "answer": "An alchemist."}',
    "invalid JSON",
])
def test_generator_refuses_unusable_model_output(retrieval_result, response):
    client = Mock()
    client.complete.return_value = response
    result = GenerationEngine(client, threshold=0.5).generate(retrieval_result)

    assert result["status"] == "refused"
    assert result["citations"] == []

@pytest.mark.parametrize("verdict, expected_status", [
    ('{"supported":true,"answer":"Petunia comforted Dudley [1].",'
     '"evidence":[{"citation_id":"[1]","quote":"Aunt Petunia comforted her son Dudley."}]}', "answered"),
    ('{"supported": true}', "refused"),
    ('{"supported": false}', "refused"),
    ('{"supported": "true"}', "refused"),
    ('invalid JSON', "refused"),
])
def test_llm_client_checks_draft_support(verdict, expected_status):
    import json
    from src.generation.llm_client import LLMClient

    client = LLMClient("test-model")
    client._chat = Mock(side_effect=[
        '{"status":"answered","answer":"Petunia comforted Dudley [1]."}',
        verdict,
    ])
    result = json.loads(client.complete(
        "Who did Petunia comfort?", "[1] Aunt Petunia comforted her son Dudley."
    ))
    assert result["status"] == expected_status
    assert client._chat.call_count == 2


def test_verifier_can_repair_citation_without_changing_answer():
    import json
    from src.generation.llm_client import LLMClient
    client = LLMClient("test-model")
    client._chat = Mock(side_effect=[
        json.dumps({"status": "answered", "answer": "Harry's wand is holly [2]."}),
        json.dumps({"supported": True, "answer": "Harry's wand is holly [1].", "evidence": [
            {"citation_id": "[1]", "quote": "Harry's wand is holly."}]}),
    ])
    result = json.loads(client.complete("What wood is Harry's wand?",
        "[1] Source: book.pdf, page 67\nHarry's wand is holly.\n\n"
        "[2] Source: book.pdf, page 68\nThe other wand is yew."))
    assert result["status"] == "answered"
    assert result["answer"] == "Harry's wand is holly [1]."
    assert result["evidence"] == [{"citation_id": "[1]", "quote": "Harry's wand is holly."}]
    verification_input = json.loads(client._chat.call_args_list[1].args[1])
    assert verification_input["sources"]["[1]:p1"]["quote"] == "Harry's wand is holly."
    assert client._chat.call_count == 2


@pytest.mark.parametrize("answer,evidence", [
    ("Harry's wand is holly [2].", [{"citation_id": "[2]", "quote": "Harry's wand is holly."}]),
    ("Harry's wand is holly [1].", [{"citation_id": "[1]", "quote": "Harry's wand is oak."}]),
    ("Harry's wand is oak [1].", [{"citation_id": "[1]", "quote": "Harry's wand is holly."}]),
    ("Harry's wand is holly [1] [2].", [{"citation_id": "[1]", "quote": "Harry's wand is holly."}]),
    ("Harry's wand is holly [9].", [{"citation_id": "[9]", "quote": "Harry's wand is holly."}]),
    ("Harry's wand is holly [1].", [{"citation_id": "[1]", "quote": ""}]),
])
def test_verification_rejects_wrong_source_quotes_and_changed_facts(answer, evidence):
    from src.generation.citations import verified_answer
    draft = "Harry's wand is holly [2]."
    assert verified_answer(draft, {"supported": True, "answer": answer, "evidence": evidence},
        {"[1]": "Harry's wand is holly.", "[2]": "The other wand is yew."}) is None


def test_source_headers_cannot_be_used_as_supporting_quotes():
    from src.generation.citations import parse_passages
    assert parse_passages("[1] Source: book.pdf, page 67\nOriginal evidence.") == {
        "[1]": "Original evidence."}


def test_citation_preview_uses_verified_quote(retrieval_result):
    import json
    retrieval_result["chunks"][0]["text"] = "An unrelated opening. Flamel was an alchemist."
    client = Mock()
    client.complete.return_value = json.dumps({"status": "answered",
        "answer": "Flamel was an alchemist [1].", "evidence": [
            {"citation_id": "[1]", "quote": "Flamel was an alchemist."}]})
    result = GenerationEngine(client, threshold=0.5).generate(retrieval_result)
    assert result["citations"][0]["snippet"] == "Flamel was an alchemist."


def test_paragraph_selection_assigns_actual_source_page():
    from src.generation.citations import evidence_options, resolve_evidence, verified_answer
    passages = {"[1]": "The other wand is yew.",
                "[2]": "Rejected trial.\n\nThe chosen wand is holly and phoenix feather."}
    options = evidence_options(passages)
    verdict = resolve_evidence({"supported": True, "answer": "The wand is holly [1].",
                                "evidence_ids": ["[2]:p2"]}, options)
    assert verdict["evidence"] == [{"citation_id": "[2]",
        "quote": "The chosen wand is holly and phoenix feather."}]
    assert verified_answer("The wand is holly.", verdict, passages) == "The wand is holly [2]."


def test_unknown_paragraph_id_is_rejected():
    from src.generation.citations import resolve_evidence
    assert resolve_evidence({"supported": True, "answer": "A fact.", "evidence_ids": ["[9]:p1"]}, {}) == {}


def test_missing_inline_citations_are_derived_from_verified_evidence():
    from src.generation.citations import verified_answer
    verdict = {"supported": True, "answer": "An alchemist.", "evidence": [
        {"citation_id": "[1]", "quote": "Flamel was an alchemist."}]}
    assert verified_answer("An alchemist.", verdict,
        {"[1]": "Flamel was an alchemist."}) == "An alchemist [1]."
