# Evaluation results (2026-11-25_20-23)

Questions: 8 · model: `qwen2.5:7b-instruct` · refusal threshold: 0.5 · verification: always · k = 5

## End-to-end

| System | Refusal rate ↑ | False refusals ↓ | Citation hit ↑ | Citation precision ↑ | Mean latency | p95 latency | Errors |
|---|---|---|---|---|---|---|---|
| full | 1.00 | 0.50 | 1.00 | 1.00 | 172439.70 ms | 211530.50 ms | 5 |

Refusal rate = refused unanswerable ÷ all unanswerable. False refusals = refused answerable ÷ all answerable.

Cost: the LLM runs locally with Ollama, so there is no API cost per question (only local compute).

### Refusal by category (full)

| Category | n | Refusal rate |
|---|---|---|
| near_miss | 1 | 1.00 |

## Threshold sweep (full pipeline)

| Threshold | Refusal rate | False refusals | Citation hit |
|---|---|---|---|
| 0.05 | 1.00 | 0.50 | 1.00 |
| 0.1 | 1.00 | 0.50 | 1.00 |
| 0.2 | 1.00 | 0.50 | 1.00 |
| 0.3 | 1.00 | 0.50 | 1.00 |
| 0.4 | 1.00 | 0.50 | 1.00 |
| 0.5 | 1.00 | 0.50 | 1.00 |
| 0.6 | 1.00 | 0.50 | 1.00 |
| 0.7 | 1.00 | 0.50 | 1.00 |
| 0.8 | 1.00 | 0.50 | 1.00 |
| 0.9 | 1.00 | 0.50 | 1.00 |
