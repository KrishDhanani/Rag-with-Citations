# RAG with Citations

Ask questions about a document collection and get an answer with citations to the retrieved passages. The pipeline combines dense search and BM25, merges their rankings with Reciprocal Rank Fusion, and reranks the candidates with a cross-encoder. Local Ollama generates the answer. The generation layer refuses when evidence is weak and checks that citation IDs refer to retrieved passages.

## Run locally

Use Python 3.12 and Ollama. Run the commands from the repository root on the `dev-3` branch, which contains the local demo.

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

## Checks and evaluation

```bash
python -m pytest tests/test_generation.py tests/test_retrieval.py -q
python -m scripts.eval_retrieval
```

The retrieval benchmark compares dense search, BM25, hybrid search, and reranking using Hit@5 and mean reciprocal rank at five. The course brief also calls for an end-to-end comparison of BM25 with a plain generation prompt against the full pipeline. Use the same questions and local model for both.

Commit a hand-labeled question set for the final public corpus. Include answerable questions with supporting passages or pages, and unanswerable questions including ones that sound relevant to the corpus. Read 50 raw outputs before scoring. Report the refusal rate as **refused unanswerable questions ÷ all unanswerable questions**, alongside false refusals on answerable questions, citation quality, latency, and cost.

## Submission and API status

The brief requires a deployed `POST /ask` that returns an answer and citations over a public corpus, plus the question set in the repository. A separate frontend is optional under the supplied brief. Confirm that the final corpus can be shared before deploying it. The bundled Harry Potter PDF is currently used for local development.

The CLI is available now. The FastAPI scaffold and full evaluation are pending. Once Dev 4 connects the route, start it locally with `python -m uvicorn src.api.main:app --reload` and try `POST /ask` through `http://127.0.0.1:8000/docs`. A deployment using this generation client needs Ollama running alongside the API on the same host.

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

## Team

| Role | Developer | Responsibility |
| --- | --- | --- |
| Dev 1 | Krish Dhanani | Parsing, chunking, and vector storage |
| Dev 2 | Areej Anjum | Hybrid retrieval, reranking, and citation tagging |
| Dev 3 | Suhaim Manna | Ollama generation, citation validation, and refusal logic |
| Dev 4 | TBD | API, deployment, and evaluation |

## Add more books

Place each book in a separate PDF file in `src/Data`, with a distinct filename. Stop the interactive question session before indexing. From the repository root, run:

```powershell
python -m scripts.ingest_data
python -m scripts.ask
```

The ingestion command scans the folder, embeds documents in batches, and updates the shared Chroma collection. Re-running it replaces the processed source's obsolete records and removes its legacy duplicates. Run one ingestion process at a time. If ingestion fails, rerun it to finish before restarting the question service. Removing a file from the folder does not remove that source from the index.

PDF chapter bookmarks are stored as chapter and section metadata where available. PDFs without chapter bookmarks use the existing text-heading fallback. Scanned PDFs require text extraction/OCR before ingestion. Restart the CLI after indexing so its BM25 index includes the added books.

Only book 1 was present when this workflow was added. Supply the remaining six book files to index the full series. Expanding the corpus changes which questions are answerable, so relabel the evaluation set against the final corpus and rerun retrieval and refusal evaluation.
