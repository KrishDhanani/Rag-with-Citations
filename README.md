# 📚 Backend RAG Engine (No UI)

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
