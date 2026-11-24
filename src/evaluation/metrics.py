"""
Pure-Python metrics for the RAG evaluation (Dev 4). No model imports, so they
are easy to unit-test.

A record is one question run through one system, as written to
raw_outputs.jsonl by scripts/evaluate_rag.py:

    {
      "id": "a01", "question": "...", "answerable": true,
      "gold_pages": [177], "gold_source": null, "category": "factual",
      "system": "full", "status": "answered" | "refused",
      "llm_status": "answered" | "refused" | null,
      "top_score": 0.93, "answer": "...",
      "citations": [{"id": "[1]", "document": "...", "page": 177, ...}],
      "retrieved": [{"id": "[1]", "document": "...", "page": 177, ...}],
      "latency_ms": 1234.5
    }

Definitions
-----------
refusal_rate          refused unanswerable / all unanswerable          (higher is better)
false_refusal_rate    refused answerable / all answerable              (lower is better)
answer_rate           answered answerable / all answerable
citation_hit_rate     answered answerable questions where at least one cited
                      page is a gold page / answered answerable       (higher is better)
citation_precision    cited passages on a gold page / all cited passages
                      (answered answerable questions only)
retrieval Hit@k, MRR  gold page among the retrieved chunks / reciprocal rank of the
                      first gold chunk (answerable questions only)
"""
from __future__ import annotations

import statistics
from collections import defaultdict


# ---- helpers -------------------------------------------------------------
def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _matches_gold(item: dict, record: dict) -> bool:
    """A cited/retrieved passage counts as correct when it is on a gold page
    (and, if gold_source is set, in the gold document)."""
    gold_pages = set(record.get("gold_pages") or [])
    if item.get("page") not in gold_pages:
        return False
    gold_source = record.get("gold_source")
    return gold_source is None or item.get("document") == gold_source


def decide(record: dict, threshold: float | None) -> str:
    """
    Final status of a 'full' record for a given refusal threshold.

    The evaluation runs the LLM on every question (llm_status) and stores the
    reranker confidence (top_score). The API refuses when top_score < threshold,
    otherwise it uses the LLM decision, so we can replay any threshold offline.
    """
    llm_status = record.get("llm_status")
    if threshold is None or llm_status is None:
        return record["status"]
    score = record.get("top_score")
    if score is None or score < threshold:
        return "refused"
    return llm_status


# ---- end-to-end metrics ----------------------------------------------------
def answer_metrics(records: list[dict], threshold: float | None = None) -> dict:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    refused_unans = sum(decide(r, threshold) == "refused" for r in unanswerable)
    refused_ans = sum(decide(r, threshold) == "refused" for r in answerable)
    answered_ans = [r for r in answerable if decide(r, threshold) == "answered"]

    hits, cited_total, cited_correct = 0, 0, 0
    for r in answered_ans:
        flags = [_matches_gold(c, r) for c in r.get("citations", [])]
        hits += any(flags)
        cited_total += len(flags)
        cited_correct += sum(flags)

    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "refusal_rate": _ratio(refused_unans, len(unanswerable)),
        "false_refusal_rate": _ratio(refused_ans, len(answerable)),
        "answer_rate": _ratio(len(answered_ans), len(answerable)),
        "citation_hit_rate": _ratio(hits, len(answered_ans)),
        "citation_precision": _ratio(cited_correct, cited_total),
    }


def refusal_by_category(records: list[dict], threshold: float | None = None) -> dict:
    """Refusal rate per category of unanswerable question (near_miss, off_topic, ...)."""
    groups = defaultdict(list)
    for r in records:
        if not r["answerable"]:
            groups[r.get("category") or "uncategorized"].append(r)
    return {
        cat: {"n": len(rs),
              "refusal_rate": _ratio(sum(decide(r, threshold) == "refused" for r in rs), len(rs))}
        for cat, rs in sorted(groups.items())
    }


def latency_stats(records: list[dict]) -> dict:
    values = sorted(r["latency_ms"] for r in records if r.get("latency_ms") is not None)
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None}
    p95_index = max(0, int(round(0.95 * len(values))) - 1)
    return {
        "mean_ms": round(statistics.mean(values), 1),
        "median_ms": round(statistics.median(values), 1),
        "p95_ms": round(values[p95_index], 1),
    }


def threshold_sweep(records: list[dict], thresholds: list[float]) -> list[dict]:
    """Refusal vs. false refusal for several thresholds ('full' records only)."""
    rows = []
    for t in thresholds:
        m = answer_metrics(records, t)
        rows.append({"threshold": t,
                     "refusal_rate": m["refusal_rate"],
                     "false_refusal_rate": m["false_refusal_rate"],
                     "citation_hit_rate": m["citation_hit_rate"]})
    return rows


# ---- retrieval-only metrics -------------------------------------------------
def retrieval_metrics(rankings: list[tuple[dict, list[dict]]], k: int) -> dict:
    """
    rankings: (question record, ranked list of {"page", "document"}) pairs.
    Only answerable questions with gold pages are scored.
    """
    scored = [(q, ranked) for q, ranked in rankings if q["answerable"] and q.get("gold_pages")]
    hits, rr = 0, 0.0
    for q, ranked in scored:
        for rank, item in enumerate(ranked[:k], start=1):
            if _matches_gold(item, q):
                hits += 1
                rr += 1.0 / rank
                break
    n = len(scored)
    return {"n": n, f"hit@{k}": _ratio(hits, n), f"mrr@{k}": round(rr / n, 4) if n else None}
