"""Tests for the RAG_VERIFY setting (when the second, verification LLM call runs)."""
import json
from unittest.mock import Mock

import pytest

from src.generation.llm_client import LLMClient, verify_mode

CONTEXT = (
    "[1] Source: book.pdf, page 177\n"
    "The only Stone currently in existence belongs to Mr. Nicolas Flamel, the noted alchemist.\n\n"
    "[2] Source: book.pdf, page 176\n"
    "Nicolas Flamel is the only known maker of the Stone."
)
GROUNDED_DRAFT = json.dumps({
    "status": "answered",
    "answer": "Nicolas Flamel is a noted alchemist and the only known maker of the Stone.",
    "evidence": [
        {"citation_id": "[1]", "quote": "belongs to Mr. Nicolas Flamel, the noted alchemist."},
        {"citation_id": "[2]", "quote": "Nicolas Flamel is the only known maker of the Stone."},
    ],
})
VERDICT_SUPPORTED = json.dumps({
    "supported": True,
    "answer": "Nicolas Flamel is a noted alchemist and the only known maker of the Stone [2].",
    "evidence_ids": ["[2]:p1"],
})


def client_with(*responses):
    client = LLMClient("test-model")
    client._chat = Mock(side_effect=list(responses))
    return client


def ask(client):
    return json.loads(client.complete("Who is Nicolas Flamel?", CONTEXT))


def test_default_mode_is_fallback(monkeypatch):
    monkeypatch.delenv("RAG_VERIFY", raising=False)
    assert verify_mode() == "fallback"
    monkeypatch.setenv("RAG_VERIFY", "nonsense")
    assert verify_mode() == "fallback"


def test_fallback_accepts_grounded_draft_with_one_call(monkeypatch):
    monkeypatch.setenv("RAG_VERIFY", "fallback")
    client = client_with(GROUNDED_DRAFT)
    result = ask(client)

    assert result["status"] == "answered"
    # labels are added from the evidence, the answer text is unchanged
    assert result["answer"] == ("Nicolas Flamel is a noted alchemist and the only known "
                                "maker of the Stone [1] [2].")
    assert client._chat.call_count == 1


@pytest.mark.parametrize("evidence", [
    [{"citation_id": "[1]", "quote": "Nicolas Flamel is a famous wizard."}],   # not in the passage
    [{"citation_id": "[2]", "quote": "the noted alchemist"}],                  # wrong passage
    [{"citation_id": "[1]", "quote": "Nicolas Flamel"}],                       # too short to count
    [{"citation_id": "[9]", "quote": "belongs to Mr. Nicolas Flamel, the noted alchemist."}],
    [],
])
def test_fallback_uses_verifier_when_draft_is_not_grounded(monkeypatch, evidence):
    monkeypatch.setenv("RAG_VERIFY", "fallback")
    draft = json.dumps({"status": "answered",
                        "answer": "Nicolas Flamel is a noted alchemist and the only known maker of the Stone.",
                        "evidence": evidence})
    client = client_with(draft, VERDICT_SUPPORTED)
    result = ask(client)

    assert client._chat.call_count == 2
    assert result["status"] == "answered"
    assert result["answer"].endswith("[2].")


def test_fallback_still_refuses_when_verifier_rejects(monkeypatch):
    monkeypatch.setenv("RAG_VERIFY", "fallback")
    draft = json.dumps({"status": "answered", "answer": "Flamel is a dragon keeper.",
                        "evidence": [{"citation_id": "[1]", "quote": "Flamel keeps dragons in Devon."}]})
    client = client_with(draft, json.dumps({"supported": False}))
    assert ask(client)["status"] == "refused"


def test_always_mode_verifies_even_grounded_drafts(monkeypatch):
    monkeypatch.setenv("RAG_VERIFY", "always")
    client = client_with(GROUNDED_DRAFT, VERDICT_SUPPORTED)
    ask(client)
    assert client._chat.call_count == 2


def test_off_mode_never_calls_verifier(monkeypatch):
    monkeypatch.setenv("RAG_VERIFY", "off")
    ungrounded = json.dumps({"status": "answered", "answer": "Flamel is a dragon keeper.",
                             "evidence": [{"citation_id": "[1]", "quote": "Flamel keeps dragons in Devon."}]})
    client = client_with(ungrounded)
    assert ask(client)["status"] == "refused"
    assert client._chat.call_count == 1


def test_model_refusal_is_unchanged(monkeypatch):
    monkeypatch.setenv("RAG_VERIFY", "fallback")
    client = client_with(json.dumps({"status": "refused", "answer": ""}))
    assert ask(client)["status"] == "refused"
    assert client._chat.call_count == 1


TYPESET_CONTEXT = (
    "[1] Source: book.pdf, page 176\n"
    "the noted alchemist Nicolas Flamel is the only known maker of the Sorcer-\n"
    "er’s Stone! He is 665 years old."
)


@pytest.mark.parametrize("quote", [
    "Nicolas Flamel, is the only known maker of the Sorcerer's Stone!",   # extra comma, straight '
    "nicolas flamel is the only known maker of the sorcerer’s stone",  # case, curly quote
])
def test_quote_matching_ignores_punctuation_case_and_line_hyphens(monkeypatch, quote):
    monkeypatch.setenv("RAG_VERIFY", "fallback")
    draft = json.dumps({"status": "answered", "answer": "Flamel made the Stone.",
                        "evidence": [{"citation_id": "[1]", "quote": quote}]})
    client = client_with(draft)
    result = json.loads(client.complete("Who made the Stone?", TYPESET_CONTEXT))
    assert result == {"status": "answered", "answer": "Flamel made the Stone [1].",
                      "evidence": [{"citation_id": "[1]", "quote": quote}]}
    assert client._chat.call_count == 1


@pytest.mark.parametrize("quote", [
    "Nicolas Flamel is the only known breaker of the Stone",   # a word changed
    "Nicolas Flamel is the only maker of the Stone",           # two separate gaps
    "only known maker of the Sorcer",                          # cut mid-word
])
def test_changed_words_still_go_to_verifier(monkeypatch, quote):
    monkeypatch.setenv("RAG_VERIFY", "fallback")
    draft = json.dumps({"status": "answered", "answer": "Flamel made the Stone.",
                        "evidence": [{"citation_id": "[1]", "quote": quote}]})
    client = client_with(draft, json.dumps({"supported": False}))
    result = json.loads(client.complete("Who made the Stone?", TYPESET_CONTEXT))
    assert result["status"] == "refused"
    assert client._chat.call_count == 2


def test_cited_label_without_quote_goes_to_verifier(monkeypatch):
    monkeypatch.setenv("RAG_VERIFY", "fallback")
    draft = json.dumps({"status": "answered",
                        "answer": "Flamel is an alchemist [1] and made the Stone [2].",
                        "evidence": [{"citation_id": "[1]",
                                      "quote": "belongs to Mr. Nicolas Flamel, the noted alchemist."}]})
    client = client_with(draft, json.dumps({"supported": False}))
    assert ask(client)["status"] == "refused"
    assert client._chat.call_count == 2


@pytest.mark.parametrize("passage, quote, expected", [
    ("Nicolas Flamel, she whispered dramatically, is the only known maker of the Stone.",
     "Nicolas Flamel is the only known maker of the Stone", True),     # dialogue tag skipped
    ("Flamel is not the only known maker of the Stone.",
     "Flamel is the only known maker of the Stone", False),            # negation may not be skipped
    ("Flamel, who lives in Devon with his wife and their old cat, made the Stone.",
     "Flamel made the Stone", False),                                  # gap longer than 5 words
    ("Flamel made the Stone in Devon.", "Flamel made the Stone", True),
    ("Flamel made the Stone in Devon.", "Flamel broke the Stone", False),
])
def test_quote_in_passage(passage, quote, expected):
    from src.generation.citations import quote_in_passage
    assert quote_in_passage(quote, passage) is expected