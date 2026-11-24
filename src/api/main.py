"""
FastAPI entry point.

Run from the repo root (Ollama must be running):
    python -m uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/ (question page) or http://127.0.0.1:8000/docs

Settings (environment variables, optional):
    RAG_LLM_MODEL           Ollama model, default qwen2.5:7b-instruct
    RAG_REFUSAL_THRESHOLD   reranker confidence below which we refuse, default 0.5
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models once at startup, not on every request.
    # Tests put a fake pipeline on app.state before starting, so nothing is loaded.
    if getattr(app.state, "pipeline", None) is None:
        from src.api.pipeline import build_pipeline
        try:
            app.state.pipeline = build_pipeline()
            app.state.startup_error = None
        except Exception as exc:          # keep the server up so /health can report it
            logger.exception("Could not load the RAG pipeline")
            app.state.pipeline = None
            app.state.startup_error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title="RAG with Citations",
    description="Ask a question about the corpus. The answer cites the retrieved "
                "passages, or the system refuses when the sources do not contain the answer.",
    lifespan=lifespan,
)
app.include_router(router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    """Small web page for asking questions (calls POST /ask)."""
    return FileResponse(STATIC_DIR / "index.html")