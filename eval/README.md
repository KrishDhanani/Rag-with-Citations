# Evaluation (Dev 4)

## Question set: `questions.jsonl`

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | Unique id, `a..` for answerable and `u..` for unanswerable |
| `question` | The question exactly as a user would ask it |
| `answerable` | `true` if the corpus contains the answer |
| `gold_pages` | PDF page numbers that contain the answer (as in the citations). Empty for unanswerable |
| `gold_source` | Optional file name, only needed when the corpus has several documents |
| `category` | `factual` for answerable. For unanswerable: `near_miss` (sounds like the corpus but is not in it), `false_premise`, `off_topic`, `injection` |
| `notes` | Why it is (un)answerable, where the answer is, who checked it |

**Current file = starter set (22 questions) for the Harry Potter demo corpus.**
The 10 answerable questions come from Dev 2's retrieval test. When the team
switches to the final public corpus, replace this file. Target: about 40
answerable and 20 or more unanswerable questions.

Rules for writing questions (from the LLM evals lecture: a golden set needs
human checking with overlap):

1. Find the answer in the PDF yourself and note the page(s). Do not take answers from the system.
2. A second team member checks each question. Write their name in `notes`.
3. Most unanswerable questions should be `near_miss`: on-topic, plausible, but not in the corpus. Those are the hard ones.
4. Do not tune the system on these questions (no special cases in the code).

## Running

```bash
python -m scripts.evaluate_rag --limit 5     # quick check that everything runs
python -m scripts.evaluate_rag               # full run, both systems
```

Results go to `eval/results/<timestamp>/`. Read `review_sample.md` (50 random
outputs) before trusting the numbers in `summary.md`.

## Metrics

- **Refusal rate** = refused unanswerable ÷ all unanswerable (the brief's "must show")
- **False-refusal rate** = refused answerable ÷ all answerable
- **Citation hit rate** = answered answerable questions where a cited page is a gold page
- **Citation precision** = cited passages on a gold page ÷ all cited passages
- **Hit@k / MRR** = retrieval only, for dense, BM25, hybrid and hybrid + rerank
- **Latency** = mean, median, p95 per question. **Cost** = 0 API cost (local Ollama)
- **Threshold sweep** = refusal vs. false refusals for thresholds 0.05 to 0.9, used to choose `RAG_REFUSAL_THRESHOLD`

The systems compared: `full` (hybrid + rerank + threshold + LLM) and
`bm25_baseline` (BM25 top-5 + the same LLM prompt, no score threshold).
