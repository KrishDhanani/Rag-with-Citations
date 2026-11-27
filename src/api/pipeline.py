"""
Glue between the API / evaluation and the team's modules (Dev 4).

Two systems are exposed with the same interface:

    full           dense + BM25 -> RRF -> cross-encoder rerank -> score threshold
                   -> grounded LLM answer with validated citations   (Dev 2 + Dev 3)
    bm25_baseline  BM25 top-k -> the same LLM prompt and citation validation.
                   BM25 scores are not calibrated, so there is no score threshold:
                   the baseline can only refuse through the LLM itself.

The baseline is what the brief asks for ("a BM25 baseline for comparison").
Both systems use the same generator, so any difference comes from retrieval.

Heavy models are imported lazily, so the API tests and the metric tests run
without torch / chromadb.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

MODES = ("full", "bm25_baseline")

DEFAULT_MODEL = os.getenv("RAG_LLM_MODEL", "qwen2.5:7b-instruct")
DEFAULT_THRESHOLD = float(os.getenv("RAG_REFUSAL_THRESHOLD", "0.5"))
# Threshold used when we want the LLM decision for every question
# (see RAGPipeline.ask(..., apply_threshold=False)). Must be > 0 for Dev 3's check.
_NO_THRESHOLD = 1e-9


@dataclass
class AskResult:
    status: str                 # "answered" | "refused"
    answer: str
    citations: list[dict]       # id, document, page, section, snippet
    mode: str
    top_score: float | None     # reranker confidence of the best chunk (full mode only)
    retrieved: list[dict]       # every chunk shown to the LLM (for evaluation)
    latency_ms: float
    llm_status: str | None = None   # LLM-only decision when the threshold was not applied


def _retrieved_view(chunks: list[dict]) -> list[dict]:
    return [
        {
            "id": c.get("citation_id"),
            "document": c.get("document"),
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
        }
        for c in chunks
    ]


class RAGPipeline:
    def __init__(self, retriever, generator_factory: Callable[[float], object],
                 *, threshold: float = DEFAULT_THRESHOLD, bm25_top_k: int | None = None):
        """
        retriever          a Dev 2 RetrievalEngine (or a fake with the same interface)
        generator_factory  threshold -> Dev 3 GenerationEngine
        """
        self.retriever = retriever
        self.threshold = threshold
        self._make_generator = generator_factory
        self.generator = generator_factory(threshold)
        self._generator_no_threshold = None
        self.bm25_top_k = bm25_top_k

    # ---- retrieval -----------------------------------------------------
    def _retrieve_bm25(self, question: str) -> dict:
        from src import config
        from src.retrieval.retriever import build_context, tag_citations

        k = self.bm25_top_k or config.FINAL_TOP_K
        hits = self.retriever.sparse.search(question, k)
        chunks = tag_citations(hits)
        return {
            "query": question.strip(),
            "chunks": chunks,
            "context": build_context(chunks),
            # No calibrated score for BM25. A score of 1.0 lets Dev 3's threshold
            # check pass, so only the LLM (and an empty BM25 result) can refuse.
            "top_score": 1.0 if chunks else 0.0,
        }

    def retrieve(self, question: str, mode: str = "full") -> dict:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        if mode == "bm25_baseline":
            return self._retrieve_bm25(question)
        return self.retriever.retrieve(question)

    # ---- full question ---------------------------------------------------
    def ask(self, question: str, mode: str = "full", *, apply_threshold: bool = True) -> AskResult:
        """
        apply_threshold=False is for the evaluation: the LLM answers every
        question regardless of the retrieval score, and the threshold is applied
        afterwards (see src/evaluation/metrics.py). This gives the exact same
        decisions as the API for any threshold, from a single run.
        """
        start = time.perf_counter()
        retrieval = self.retrieve(question, mode)

        if apply_threshold or mode != "full":
            generator = self.generator
        else:
            if self._generator_no_threshold is None:
                self._generator_no_threshold = self._make_generator(_NO_THRESHOLD)
            generator = self._generator_no_threshold

        result = generator.generate(retrieval)
        latency_ms = (time.perf_counter() - start) * 1000

        return AskResult(
            status=result["status"],
            answer=result["answer"],
            citations=result["citations"],
            mode=mode,
            top_score=retrieval.get("top_score") if mode == "full" else None,
            retrieved=_retrieved_view(retrieval["chunks"]),
            latency_ms=round(latency_ms, 1),
            llm_status=None if apply_threshold or mode != "full" else result["status"],
        )


def build_pipeline(model: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD) -> RAGPipeline:
    """Load the real models once (embedding, cross-encoder, BM25, Ollama client)."""
    from src.generation.generator import GenerationEngine
    from src.generation.llm_client import LLMClient
    from src.retrieval.retriever import RetrievalEngine

    retriever = RetrievalEngine()
    client = LLMClient.from_ollama(model)
    return RAGPipeline(
        retriever,
        lambda t: GenerationEngine(client, threshold=t),
        threshold=threshold,
    )
