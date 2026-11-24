# Evaluation results (2026-11-20_19-56)

Questions: 22 · model: `qwen2.5:7b-instruct` · refusal threshold: 0.5 · k = 5

## End-to-end

| System | Refusal rate ↑ | False refusals ↓ | Citation hit ↑ | Citation precision ↑ | Mean latency | p95 latency | Errors |
|---|---|---|---|---|---|---|---|
| full | 1.00 | 0.00 | 0.90 | 0.64 | 25774.50 ms | 30659.40 ms | 0 |
| bm25_baseline | 0.92 | 0.00 | 0.60 | 0.43 | 21428.00 ms | 29450.60 ms | 0 |

Refusal rate = refused unanswerable ÷ all unanswerable. False refusals = refused answerable ÷ all answerable.

Cost: the LLM runs locally with Ollama, so there is no API cost per question (only local compute).

### Refusal by category (full)

| Category | n | Refusal rate |
|---|---|---|
| false_premise | 2 | 1.00 |
| injection | 1 | 1.00 |
| near_miss | 7 | 1.00 |
| off_topic | 2 | 1.00 |

### Refusal by category (bm25_baseline)

| Category | n | Refusal rate |
|---|---|---|
| false_premise | 2 | 1.00 |
| injection | 1 | 1.00 |
| near_miss | 7 | 0.86 |
| off_topic | 2 | 1.00 |

## Threshold sweep (full pipeline)

| Threshold | Refusal rate | False refusals | Citation hit |
|---|---|---|---|
| 0.05 | 1.00 | 0.00 | 0.90 |
| 0.1 | 1.00 | 0.00 | 0.90 |
| 0.2 | 1.00 | 0.00 | 0.90 |
| 0.3 | 1.00 | 0.00 | 0.90 |
| 0.4 | 1.00 | 0.00 | 0.90 |
| 0.5 | 1.00 | 0.00 | 0.90 |
| 0.6 | 1.00 | 0.00 | 0.90 |
| 0.7 | 1.00 | 0.00 | 0.90 |
| 0.8 | 1.00 | 0.00 | 0.90 |
| 0.9 | 1.00 | 0.10 | 0.89 |

## Retrieval only (answerable questions)

| Retriever | Hit@5 | MRR@5 |
|---|---|---|
| dense | 0.90 | 0.49 |
| bm25 | 0.70 | 0.35 |
| hybrid (RRF) | 0.90 | 0.41 |
| hybrid + rerank | 0.90 | 0.66 |
