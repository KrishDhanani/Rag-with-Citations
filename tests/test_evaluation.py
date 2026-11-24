"""Tests for the evaluation metrics (pure Python, no models)."""
import json

import pytest

from scripts.evaluate_rag import load_questions
from src.evaluation.metrics import (
    answer_metrics,
    decide,
    latency_stats,
    refusal_by_category,
    retrieval_metrics,
    threshold_sweep,
)


def rec(answerable, status, pages=(), gold=(10,), category=None, top=None, llm=None, ms=100.0):
    return {"answerable": answerable, "status": status, "gold_pages": list(gold) if answerable else [],
            "gold_source": None, "category": category or ("factual" if answerable else "near_miss"),
            "citations": [{"id": f"[{i}]", "page": p, "document": "d.pdf"}
                          for i, p in enumerate(pages, 1)],
            "top_score": top, "llm_status": llm, "latency_ms": ms}


def test_answer_metrics_basic():
    records = [
        rec(True, "answered", pages=[10]),          # correct citation
        rec(True, "answered", pages=[3, 10]),       # one of two citations correct
        rec(True, "answered", pages=[3]),           # wrong citation
        rec(True, "refused"),                       # false refusal
        rec(False, "refused"),
        rec(False, "refused"),
        rec(False, "answered", pages=[5]),          # hallucinated answer
    ]
    m = answer_metrics(records)

    assert m["n_answerable"] == 4 and m["n_unanswerable"] == 3
    assert m["refusal_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["false_refusal_rate"] == 0.25
    assert m["answer_rate"] == 0.75
    assert m["citation_hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["citation_precision"] == 0.5        # 2 correct of 4 cited


def test_metrics_empty_groups_are_none():
    m = answer_metrics([rec(True, "answered", pages=[10])])
    assert m["refusal_rate"] is None
    assert m["citation_hit_rate"] == 1.0


def test_gold_source_must_match():
    r = rec(True, "answered", pages=[10])
    r["gold_source"] = "other.pdf"
    assert answer_metrics([r])["citation_hit_rate"] == 0.0


def test_decide_replays_threshold():
    r = rec(True, "answered", top=0.4, llm="answered")
    assert decide(r, 0.3) == "answered"
    assert decide(r, 0.5) == "refused"
    assert decide(rec(True, "refused", top=0.9, llm="refused"), 0.1) == "refused"
    assert decide(rec(True, "answered"), 0.9) == "answered"      # baseline: no llm_status


def test_threshold_sweep_is_monotonic():
    records = [rec(False, "x", top=s, llm="answered") for s in (0.1, 0.3, 0.6, 0.9)] + \
              [rec(True, "x", pages=[10], top=s, llm="answered") for s in (0.2, 0.7, 0.95)]
    sweep = threshold_sweep(records, [0.05, 0.5, 0.99])

    assert [r["refusal_rate"] for r in sweep] == [0.0, 0.5, 1.0]
    assert [r["false_refusal_rate"] for r in sweep] == [0.0, pytest.approx(1 / 3, abs=1e-4), 1.0]


def test_refusal_by_category():
    records = [rec(False, "refused", category="near_miss"),
               rec(False, "answered", category="near_miss"),
               rec(False, "refused", category="off_topic")]
    out = refusal_by_category(records)
    assert out == {"near_miss": {"n": 2, "refusal_rate": 0.5},
                   "off_topic": {"n": 1, "refusal_rate": 1.0}}


def test_retrieval_metrics():
    q1 = {"answerable": True, "gold_pages": [5], "gold_source": None}
    q2 = {"answerable": True, "gold_pages": [9], "gold_source": None}
    q3 = {"answerable": False, "gold_pages": []}
    rankings = [(q1, [{"page": 1}, {"page": 5}]),
                (q2, [{"page": 2}, {"page": 3}]),
                (q3, [{"page": 9}])]
    m = retrieval_metrics(rankings, k=5)
    assert m == {"n": 2, "hit@5": 0.5, "mrr@5": 0.25}


def test_latency_stats():
    stats = latency_stats([{"latency_ms": v} for v in range(1, 101)])
    assert stats["median_ms"] == 50.5 and stats["p95_ms"] == 95


def test_load_questions(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(json.dumps({"id": "a1", "question": "q?", "answerable": True, "gold_pages": [1]})
                 + "\n\n", encoding="utf-8")
    assert load_questions(f)[0]["id"] == "a1"

    f.write_text('{"id": "a1", "question": "q?"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="answerable"):
        load_questions(f)


def test_repo_question_set_is_valid():
    questions = load_questions(__import__("pathlib").Path("eval/questions.jsonl"))
    assert any(q["answerable"] for q in questions)
    assert any(not q["answerable"] for q in questions)
