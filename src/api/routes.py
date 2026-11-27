import logging

import requests
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import HealthResponse, QuestionRequest, QuestionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_pipeline(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        error = getattr(request.app.state, "startup_error", None)
        raise HTTPException(503, f"Pipeline not available: {error or 'still loading'}")
    return pipeline


@router.post("/ask", response_model=QuestionResponse)
def ask_question(payload: QuestionRequest, request: Request):
    """Answer a question from the corpus with citations, or refuse."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(422, "question must not be empty")

    pipeline = _get_pipeline(request)
    try:
        result = pipeline.ask(question, payload.mode)
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, "The LLM server (Ollama) is not reachable. Is Ollama running?")
    except requests.exceptions.Timeout:
        raise HTTPException(504, "The LLM took too long to answer.")
    except Exception:
        logger.exception("ask failed for question %r", question)
        raise HTTPException(500, "Internal error while answering the question.")

    return QuestionResponse(
        answer=result.answer,
        status=result.status,
        citations=result.citations,
        mode=result.mode,
        latency_ms=result.latency_ms,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    if getattr(request.app.state, "pipeline", None) is not None:
        return HealthResponse(status="ok")
    error = getattr(request.app.state, "startup_error", None)
    return HealthResponse(status="error" if error else "loading", detail=error)
