# 📚 RAG with Citation

Modular RAG system with hybrid search (Dense + BM25), strict citation grounding, refusal thresholding, and automated evaluation.

## 👥 Team

| Role | Developer | Responsibility |
| :--- | :--- | :--- |
| Dev 1: Ingestion & Storage | Krish Dhanani | Data parsing, chunking, vector DB setup |
| Dev 2: Hybrid Retrieval Engine | **Areej Anjum** | Dense + BM25 search, RRF fusion, cross-encoder reranking, citation tagging |
| Dev 3: Generation & Safety | TBD | LLM prompting, citation formatting, refusal logic |
| Dev 4: API & Evaluation | TBD | POST /ask API, evaluation suite, refusal rate metrics |

## 🚀 Quickstart
1. Clone the repo and create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. `.env` add your API keys.
4. Run the API:
   ```bash
   uvicorn src.api.main:app --reload
   ```

## 📂 Repository Structure

```text
Rag-with-Citations/
├── Data/
│   ├── Any File Type          # Original corpus files (PDFs, Markdown, TXT)
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Global settings, paths, DB connections & API keys
│   │
│   ├── ingestion/             
│   │   ├── __init__.py
|   |   |── ingest_data.py     # Main file to run whole ingestion process at once 
│   │   ├── parser.py          # Document loading & markdown/text extraction
│   │   ├── chunker.py         # Chunking and Re-Section the metadata 
│   │   ├── vector_store.py    # Vector DB initialization & dense embedding loading
│   │
│   ├── retrieval/             # --- DEV 2 WORKSPACE ---
│   │   ├── __init__.py
│   │   ├── dense_search.py    # Vector DB query execution
│   │   ├── sparse_search.py   # BM25 baseline search execution
│   │   ├── hybrid_fusion.py   # RRF (Reciprocal Rank Fusion) algorithm
│   │   ├── reranker.py        # Cross-encoder reranking engine
│   │   └── retriever.py       # Main RetrievalEngine orchestrator
│   │
│   ├── generation/            # --- DEV 3 WORKSPACE ---
│   │   ├── __init__.py
│   │   ├── llm_client.py      # OpenAI/Claude/Ollama client interface
│   │   ├── prompts.py         # System & RAG prompt templates with citation constraints
│   │   ├── refusal.py         # Threshold checking & refusal decision logic
│   │   └── generator.py       # Main GenerationEngine orchestrator
│   │
│   └── api/                   # --- DEV 4 WORKSPACE ---
│       ├── __init__.py
│       ├── main.py            # FastAPI application entry point
│       ├── schemas.py         # Pydantic models for POST /ask request/response
│       └── routes.py          # /ask endpoint route logic
│
├── tests/                     # Integration tests
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_api.py
│
├── scripts/                   # Run & Evaluation Scripts
│   ├── ingest_data.py         # Dev 1: Pipeline execution script to index corpus
│   ├── eval_retrieval.py      # Dev 2: Retrieval ablation (dense vs BM25 vs hybrid vs rerank)
│   └── evaluate_rag.py        # Dev 4: Evaluates metrics & unanswerable refusal rate
│
├── .env.example               # Template for environment variables
├── Dockerfile                 # Container setup
├── README.md                  # Project overview & run instructions
└── requirements.txt           # Project dependencies
```

## 🔎 Retrieval (Dev 2 - Areej Anjum)

Pipeline: dense search (MiniLM, Chroma) + BM25 → Reciprocal Rank Fusion → cross-encoder reranking → citation tagging.

```python
from src.retrieval.retriever import RetrievalEngine

engine = RetrievalEngine()                 # load once, models are cached
result = engine.retrieve("Who is Nicolas Flamel?")

result["context"]     # passages tagged [1], [2], ... ready for the prompt
result["chunks"]      # citation_id, document, page, section, snippet, scores
result["top_score"]   # 0..1 confidence of the best passage (for the refusal threshold)
```

Run from the repo root:

```bash
python -m src.retrieval.retriever "Who gave Harry the invisibility cloak?"   # manual check
python -m pytest tests/test_retrieval.py                                     # unit tests
python -m scripts.eval_retrieval                                             # Hit@5 / MRR comparison
```

All settings (models, top-k values, RRF constant) are in `src/config.py`.
