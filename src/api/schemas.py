from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          examples=["Who is Nicolas Flamel?"])
    mode: Literal["full", "bm25_baseline"] = Field(
        "full",
        description="'full' = hybrid retrieval + reranker + refusal threshold; "
                    "'bm25_baseline' = BM25 only, for comparison.",
    )


class Citation(BaseModel):
    id: str                          # "[1]", matches the label in the answer text
    document: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str


class QuestionResponse(BaseModel):
    answer: str
    status: Literal["answered", "refused"]
    citations: List[Citation]
    mode: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    detail: Optional[str] = None
