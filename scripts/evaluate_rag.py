"""
End-to-end evaluation of the RAG system (Dev 4).

What it measures, for the full pipeline and for the BM25 baseline:
  * refusal rate on unanswerable questions       (the "must show" metric)
  * false-refusal rate on answerable questions
  * citation hit rate / precision (cited page is a gold page)
  * latency (mean, median, p95)
  * retrieval Hit@k and MRR for dense, BM25, hybrid and hybrid + rerank
  * a threshold sweep, to choose the refusal threshold with data

Run from the repo root (Ollama must be running):

    python -m scripts.evaluate_rag                         # everything
    python -m scripts.evaluate_rag --limit 5               # quick smoke test
    python -m scripts.evaluate_rag --systems full          # only the full pipeline
    python -m scripts.evaluate_rag --questions eval/questions.jsonl --threshold 0.5

Output goes to eval/results/<timestamp>/:
    raw_outputs.jsonl   every answer, citation and score (one line per question+system)
    review_sample.md    50 outputs to read and judge by hand BEFORE trusting the numbers
    summary.json        all metrics
    summary.md          tables for the report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from src.evaluation.metrics import (
    answer_metrics,
    decide,
    latency_stats,
    refusal_by_category,
    retrieval_metrics,
    threshold_sweep,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = REPO_ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REQUIRED_FIELDS = {"id", "question", "answerable"}
REFUSAL_TEXT = "(refused: retrieval score below threshold)"


# ---- question set -----------------------------------------------------------
def load_questions(path: Path) -> list[dict]:
    questions, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            q = json.loads(line)
            missing = REQUIRED_FIELDS - q.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} is missing {sorted(missing)}")
            if q["id"] in seen:
                raise ValueError(f"{path}:{line_no} duplicate id {q['id']!r}")
            if q["answerable"] and not q.get("gold_pages"):
                print(f"warning: {q['id']} is answerable but has no gold_pages "
                      "(it will not count for citation/retrieval metrics)", file=sys.stderr)
            seen.add(q["id"])
            questions.append(q)
    return questions


def balanced_subset(questions: list[dict], n: int) -> list[dict]:
    """First n questions, taking answerable and unanswerable ones in turn, so a
    quick run still measures both refusals and false refusals."""
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    picked = []
    while len(picked) < n and (answerable or unanswerable):
        for group in (answerable, unanswerable):
            if group and len(picked) < n:
                picked.append(group.pop(0))
    return picked


# ---- running the systems ----------------------------------------------------
def run_system(pipeline, questions: list[dict], mode: str, threshold: float) -> list[dict]:
    records = []
    for i, q in enumerate(questions, start=1):
        base = {k: q.get(k) for k in ("id", "question", "answerable", "gold_pages",
                                      "gold_source", "category")}
        try:
            # full mode: the LLM answers every question, the threshold is applied
            # afterwards with decide() -> identical to the API for any threshold
            res = pipeline.ask(q["question"], mode, apply_threshold=False)
            record = {**base, "system": mode, "status": res.status,
                      "llm_status": res.llm_status, "top_score": res.top_score,
                      "answer": res.answer, "citations": res.citations,
                      "retrieved": res.retrieved, "latency_ms": res.latency_ms}
            if res.llm_status is not None:
                record["status"] = decide(record, threshold)
                if record["status"] == "refused" and res.llm_status == "answered":
                    # refused by the score threshold: what the API would return
                    record["llm_answer"] = res.answer
                    record["answer"] = REFUSAL_TEXT
                    record["citations"] = []
        except Exception as exc:
            record = {**base, "system": mode, "status": "error",
                      "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        print(f"  [{mode}] {i}/{len(questions)} {q['id']}: {record['status']}", flush=True)
    return records


def run_retrieval_ablation(pipeline, questions: list[dict], k: int) -> dict:
    """Hit@k / MRR for the four retrieval variants (answerable questions only)."""
    from src import config
    from src.retrieval.hybrid_fusion import reciprocal_rank_fusion

    r = pipeline.retriever
    variants = {
        "dense": lambda q: r.dense.search(q, config.DENSE_TOP_K),
        "bm25": lambda q: r.sparse.search(q, config.SPARSE_TOP_K),
        "hybrid (RRF)": lambda q: reciprocal_rank_fusion(
            r.dense.search(q, config.DENSE_TOP_K), r.sparse.search(q, config.SPARSE_TOP_K)),
        "hybrid + rerank": lambda q: r.reranker.rerank(
            q, reciprocal_rank_fusion(r.dense.search(q, config.DENSE_TOP_K),
                                      r.sparse.search(q, config.SPARSE_TOP_K))[:config.RERANK_TOP_N],
            k),
    }
    scored = [q for q in questions if q["answerable"] and q.get("gold_pages")]
    results = {}
    for name, search in variants.items():
        rankings = []
        for q in scored:
            hits = search(q["question"])
            rankings.append((q, [{"page": h["metadata"].get("page"),
                                  "document": h["metadata"].get("source")} for h in hits]))
        results[name] = retrieval_metrics(rankings, k)
    return results


# ---- reports -----------------------------------------------------------------
def fmt(value) -> str:
    if value is None:
        return "–"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def write_review_sample(records: list[dict], path: Path, n: int = 50, seed: int = 42):
    """Random sample of outputs for manual reading (as required before scoring)."""
    pool = [r for r in records if r["status"] != "error"]
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    lines = ["# Manual review sample", "",
             "Read every answer and mark it. Check that the answer is correct AND that "
             "the cited page really supports it.", ""]
    for i, r in enumerate(sample, start=1):
        expected = (f"answerable, gold pages {r.get('gold_pages')}" if r["answerable"]
                    else f"UNANSWERABLE ({r.get('category')})")
        cited = ", ".join(f"{c['id']} p.{c.get('page')}" for c in r.get("citations", [])) or "–"
        lines += [
            f"## {i}. `{r['id']}` · system: {r['system']}",
            f"**Q:** {r['question']}  ",
            f"**Expected:** {expected}  ",
            f"**Status:** {r['status']} · top_score: {fmt(r.get('top_score'))} · cited: {cited}",
            "",
            f"> {r.get('answer') or '(empty)'}",
            "",
            "Verdict: [ ] correct  [ ] wrong answer  [ ] wrong/unsupported citation  "
            "[ ] should have refused  [ ] should have answered",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(summary: dict, path: Path):
    s = summary
    lines = [f"# Evaluation results ({s['run_at']})", "",
             f"Questions: {s['n_questions']} · model: `{s['model']}` · "
             f"refusal threshold: {s['threshold']} · verification: {s.get('verify_mode', '–')} · k = {s['k']}", ""]

    lines += ["## End-to-end", "",
              "| System | Refusal rate ↑ | False refusals ↓ | Citation hit ↑ | "
              "Citation precision ↑ | Mean latency | p95 latency | Errors |",
              "|---|---|---|---|---|---|---|---|"]
    for name, m in s["systems"].items():
        a, lat = m["answers"], m["latency"]
        lines.append(f"| {name} | {fmt(a['refusal_rate'])} | {fmt(a['false_refusal_rate'])} | "
                     f"{fmt(a['citation_hit_rate'])} | {fmt(a['citation_precision'])} | "
                     f"{fmt(lat['mean_ms'])} ms | {fmt(lat['p95_ms'])} ms | {m['n_errors']} |")
    lines += ["", "Refusal rate = refused unanswerable ÷ all unanswerable. "
              "False refusals = refused answerable ÷ all answerable.", "",
              "Cost: the LLM runs locally with Ollama, so there is no API cost per question "
              "(only local compute).", ""]

    for name, m in s["systems"].items():
        lines += [f"### Refusal by category ({name})", "", "| Category | n | Refusal rate |",
                  "|---|---|---|"]
        lines += [f"| {cat} | {v['n']} | {fmt(v['refusal_rate'])} |"
                  for cat, v in m["refusal_by_category"].items()]
        lines.append("")

    if s.get("threshold_sweep"):
        lines += ["## Threshold sweep (full pipeline)", "",
                  "| Threshold | Refusal rate | False refusals | Citation hit |", "|---|---|---|---|"]
        lines += [f"| {r['threshold']} | {fmt(r['refusal_rate'])} | "
                  f"{fmt(r['false_refusal_rate'])} | {fmt(r['citation_hit_rate'])} |"
                  for r in s["threshold_sweep"]]
        lines.append("")

    if s.get("retrieval"):
        k = s["k"]
        lines += ["## Retrieval only (answerable questions)", "",
                  f"| Retriever | Hit@{k} | MRR@{k} |", "|---|---|---|"]
        lines += [f"| {name} | {fmt(m[f'hit@{k}'])} | {fmt(m[f'mrr@{k}'])} |"
                  for name, m in s["retrieval"].items()]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---- main ----------------------------------------------------------------------
def main(argv=None):
    from src.api.pipeline import DEFAULT_MODEL, DEFAULT_THRESHOLD, MODES
    from src.generation.llm_client import verify_mode

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--systems", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--k", type=int, default=5, help="cut-off for Hit@k / MRR")
    parser.add_argument("--limit", type=int,
                        help="only N questions, half answerable and half unanswerable (quick test)")
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="skip the retrieval-only ablation")
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    if args.limit:
        questions = balanced_subset(questions, args.limit)
    n_ans = sum(q["answerable"] for q in questions)
    print(f"{len(questions)} questions ({n_ans} answerable, {len(questions) - n_ans} unanswerable)")

    print("Loading models ...")
    from src.api.pipeline import build_pipeline
    pipeline = build_pipeline(model=args.model, threshold=args.threshold)

    run_at = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = RESULTS_DIR / run_at
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records, systems = [], {}
    for mode in args.systems:
        print(f"\nRunning {mode} ...")
        t0 = time.perf_counter()
        records = run_system(pipeline, questions, mode, args.threshold)
        ok = [r for r in records if r["status"] != "error"]
        systems[mode] = {
            "answers": answer_metrics(ok),
            "refusal_by_category": refusal_by_category(ok),
            "latency": latency_stats(ok),
            "n_errors": len(records) - len(ok),
            "wall_time_s": round(time.perf_counter() - t0, 1),
        }
        all_records += records

    summary = {
        "run_at": run_at, "model": args.model, "threshold": args.threshold, "k": args.k,
        "verify_mode": verify_mode(),
        "questions_file": str(args.questions), "n_questions": len(questions),
        "systems": systems,
    }
    full_ok = [r for r in all_records if r["system"] == "full" and r["status"] != "error"]
    if full_ok:
        summary["threshold_sweep"] = threshold_sweep(full_ok, SWEEP)
    if not args.skip_retrieval:
        print("\nRetrieval ablation ...")
        summary["retrieval"] = run_retrieval_ablation(pipeline, questions, args.k)

    with open(out_dir / "raw_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    write_summary_md(summary, out_dir / "summary.md")
    write_review_sample(all_records, out_dir / "review_sample.md")

    print(f"\nDone. Results in {out_dir}")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()