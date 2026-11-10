# 📚 RAG with Citation

Modular RAG system with hybrid search (Dense + BM25), strict citation grounding, refusal thresholding, and automated evaluation.

## 👥 Team Roles & Responsibilities
See `roles_table.md` for team division.

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
3. Copy `.env.example` to `.env` and add your API keys.
4. Run the API:
   ```bash
   uvicorn src.api.main:app --reload
   ```

## 📂 Repository Structure

```text
rag-project/
├── data/
│   ├── raw/                   # Original corpus files (PDFs, Markdown, TXT)
│   ├── processed/             # Cleaned & chunked JSON/JSONL datasets
│   └── eval_questions.json    # Benchmark dataset (answerable + unanswerable)
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Global settings, paths, DB connections & API keys
│   │
│   ├── ingestion/             # --- DEV 1 WORKSPACE ---
│   │   ├── __init__.py
│   │   ├── parser.py          # Document loading & markdown/text extraction
│   │   ├── chunker.py         # Fixed vs semantic chunking strategies
│   │   ├── vector_store.py    # Vector DB initialization & dense embedding loading
│   │   └── bm25_indexer.py    # BM25 sparse index builder & disk storage
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
│   └── evaluate_rag.py        # Dev 4: Evaluates metrics & unanswerable refusal rate
│
├── .env.example               # Template for environment variables
├── Dockerfile                 # Container setup
├── README.md                  # Project overview & run instructions
└── requirements.txt           # Project dependencies
