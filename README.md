# RAG with Citations

Local Harry Potter Q&A with citations to the original PDF pages. The current corpus contains only *Harry Potter and the Sorcerer's Stone*, so questions requiring later books should be refused.

The pipeline searches paragraph chunks with dense embeddings and BM25, merges rankings with Reciprocal Rank Fusion, and reranks candidates with a cross-encoder. Cached, source-extracted page context enriches both search indexes. Local Ollama generates and verifies the answer; selected evidence paragraphs determine the citation text and page numbers.

## Run locally

Use Python 3.12 and Ollama. Run the commands from the repository root on the `para-chunk` branch:

```bash
git switch para-chunk
```

On Windows, in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, create and activate the environment with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open the Ollama app, or run `ollama serve` in a separate terminal if the server isn't running. Pull the model once, then start the demo:

```bash
ollama pull qwen2.5:7b-instruct
python -m scripts.ask
```

Type a question and press Enter. Type `exit` to quit. For a single question, run:

```bash
python -m scripts.ask "Who is Nicolas Flamel?"
```

Flamel should return an answer with PDF citations. Try “What is Polyjuice Potion?” for an expected refusal on the current demo corpus. The demo uses the bundled Chroma index in `src/Vectorstore` and the example PDF in `src/Data`. It needs no OpenAI API key. The first run may download the retrieval models from Hugging Face. The CLI supports `--model` and `--threshold`. The default refusal cutoff of `0.5` is provisional.

The bundled index has been rebuilt with 829 paragraph chunks across 249
text-bearing PDF pages. Start asking questions directly; ingestion is needed only
when changing the corpus or indexing settings. Exit and restart the question
session after updating code or rebuilding the index.

## Checks and evaluation

```bash
python -m pytest tests -q
python -m scripts.eval_retrieval
# Requires Ollama; checks answers and citation pages against the current Book 1 index.
python -m scripts.check_answers
```

Validation on this branch: **59 automated tests passed**, and all three live
regression checks passed with `qwen2.5:7b-instruct`:

| Question | Verified result |
| --- | --- |
| Who is Harry's mother? | Lily Potter; supporting PDF pages 42 and 10 |
| What is Harry's wand made of? | Holly and phoenix feather; PDF page 67 |
| What is Polyjuice Potion? | Refused because the current corpus lacks evidence |

These three live checks cover the reported regressions, not overall universe-wide
accuracy. Mocked tests verify program behavior; they do not establish whether the
local model correctly interprets every passage.

The retrieval benchmark compares dense search, BM25, hybrid search, and reranking using Hit@5 and mean reciprocal rank at five. The course brief also calls for an end-to-end comparison of BM25 with a plain generation prompt against the full pipeline. Use the same questions and local model for both.

Commit a hand-labeled question set for the final public corpus. Include answerable questions with supporting passages or pages, and unanswerable questions including ones that sound relevant to the corpus. Read 50 raw outputs before scoring. Report the refusal rate as **refused unanswerable questions ÷ all unanswerable questions**, alongside false refusals on answerable questions, citation quality, latency, and cost.

## Submission and API status

The brief requires a deployed `POST /ask` that returns an answer and citations over a public corpus, plus the question set in the repository. A separate frontend is optional under the supplied brief. Confirm that the final corpus can be shared before deploying it. The bundled Harry Potter PDF is currently used for local development.

Deployment is still open: the API needs Ollama running on the same host.

## API (Dev 4)

Start the API from the repository root while Ollama is running:

```bash
python -m uvicorn src.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try it in the browser, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Who is Nicolas Flamel?"}'
```

Response:

```json
{
  "answer": "Nicolas Flamel is ... [1].",
  "status": "answered",
  "citations": [{"id": "[1]", "document": "...pdf", "page": 177, "section": "...", "snippet": "..."}],
  "mode": "full",
  "latency_ms": 2140.5
}
```

`status` is `answered` or `refused`. Refusals return an empty citation list. Set `"mode": "bm25_baseline"` to answer with BM25 retrieval only, for comparison. `GET /health` reports whether the models loaded. The environment variables `RAG_LLM_MODEL` and `RAG_REFUSAL_THRESHOLD` (default `0.5`) override the model and the refusal cutoff.

## Evaluation (Dev 4)

```bash
python -m scripts.evaluate_rag --limit 5   # quick check
python -m scripts.evaluate_rag             # full run
```

This runs every question in `eval/questions.jsonl` through the full pipeline and the BM25 baseline. It reports the refusal rate, false refusals, citation accuracy, latency, a threshold sweep, and retrieval Hit@5 and MRR. Results go to `eval/results/<timestamp>/`. Read `review_sample.md` before trusting `summary.md`. See `eval/README.md` for the question format and metric definitions.

For the API integration, create the retrieval and generation engines once and pass each retrieval result to generation:

```python
from src.generation.generator import GenerationEngine
from src.generation.llm_client import LLMClient
from src.retrieval.retriever import RetrievalEngine

retriever = RetrievalEngine()
generator = GenerationEngine(LLMClient.from_ollama("qwen2.5:7b-instruct"), threshold=0.5)

result = generator.generate(retriever.retrieve("Who is Nicolas Flamel?"))
```

The result contains `answer`, `status` (`answered` or `refused`), and `citations`. Each citation includes its ID, document, PDF page, section, and snippet. Refusals return an empty citation list.

Relationship retrieval keeps the original question alongside the broader parent
query. Name-based follow-up passages supplement the original evidence and are
reranked together; the refusal score comes from the final selected passage.
Family name hints come from explicit parent/child constructions in source pages;
pairs of siblings sharing a surname are not treated as parents. The existing
verification call selects original paragraph IDs instead of rewriting quotations.
The application maps those IDs to exact text and source pages, attaches citation
labels, and rejects unknown IDs and answer rewrites. Citation snippets show the
selected original text. Semantic support still depends on the verifier and needs
evaluation beyond unit tests. These retrieval and generation changes do not
require reindexing.

Run `python -m scripts.check_answers` separately for live Ollama checks of Harry's
mother, his wand materials and page citation, and the expected Book 1 Polyjuice
refusal. This reads the existing index without running ingestion. The refusal
expectation applies only to the current Book 1 corpus; update it if adding books.

## Team

| Role | Developer | Responsibility |
| --- | --- | --- |
| Dev 1 | Krish Dhanani | Parsing, chunking, and vector storage |
| Dev 2 | Areej Anjum | Hybrid retrieval, reranking, and citation tagging |
| Dev 3 | Suhaim Manna | Ollama generation, citation validation, and refusal logic |
| Dev 4 | Muhammad Talha Khan | API, deployment, and evaluation |

## Add more books

Place each book in a separate PDF file in `src/Data`, with a distinct filename. Stop the interactive question session before indexing. From the repository root, run:

```powershell
ollama pull qwen3:1.7b
python -m scripts.ingest_data
python -m scripts.ask
```

The ingestion command scans the folder, embeds documents in batches, and updates the shared Chroma collection. Re-running it replaces the processed source's obsolete records and removes its legacy duplicates. Run one ingestion process at a time. If ingestion fails, rerun it to finish before restarting the question service. Removing a file from the folder does not remove that source from the index.

PDF chapter bookmarks are stored as chapter and section metadata where available. PDFs without chapter bookmarks use the existing text-heading fallback. Scanned PDFs require text extraction/OCR before ingestion. Restart the CLI after indexing so its BM25 index includes the added books.

### Paragraph chunks and contextual search

Ingestion now preserves PDF text blocks as paragraphs, groups short paragraphs
into chunks of up to 750 characters, and overlaps whole paragraphs up to 150
characters. Oversized paragraphs use sentence/word splitting. Chunks stay within
one PDF page so citations retain exact page numbers.

By default, local Ollama selects short context phrases for each page using that
page and its immediate neighbors within the same chapter. Each selected phrase
must occur in the source text; invented phrases are discarded. If none match,
an original page excerpt is used instead. All chunks on the page share this
context. This is page-level extractive contextual enrichment, rather than a
separate generated summary for every chunk. Context is selected only at ingestion,
cached under `.cache/contextual`, and prepended for embedding and BM25 indexing.
Original passages alone are stored as evidence and used for reranking, answers,
and citations. Selected context can still influence retrieval incorrectly;
evaluate its quality on your question set before claiming an accuracy gain.

```powershell
# Ollama must be running. Resume an interrupted run with the same command.
ollama pull qwen3:1.7b
python -m scripts.ingest_data
# Override the local model used for context generation.
python -m scripts.ingest_data --context-model qwen2.5:7b-instruct
# Ablation: paragraph chunks with no generated contextual descriptions.
python -m scripts.ingest_data --no-context
```

The first contextual run is slower; unchanged pages reuse cached descriptions.
Changes to the prompt, model name, or input text invalidate the relevant cache.
Incomplete or malformed model output falls back to an original page excerpt.
Connection failures stop ingestion before replacing that source's index.
Reindexing replaces the old chunks for each processed source; restart the question
session afterward. After reranking paragraph chunks, matches on the same page are
deduplicated and the generator receives the complete original page dialogue with
repeated paragraphs removed. This retains narrative context for trials, choices,
and relationships. Legacy indexes remain readable, but must be rebuilt to benefit
from the new chunking and embeddings.

Run the focused checks with
`python -m pytest tests/test_ingestion.py tests/test_retrieval.py tests/test_generation.py -q`.

Only book 1 was present when this workflow was added. Supply the remaining six book files to index the full series. Expanding the corpus changes which questions are answerable, so relabel the evaluation set against the final corpus and rerun retrieval and refusal evaluation.
