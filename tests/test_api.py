"""
Tests for POST /ask and the pipeline glue. The real models are replaced by
fakes, so nothing is downloaded and Ollama is not needed.

Run from the repo root:  python -m pytest tests/test_api.py
"""
import pytest
import requests
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.pipeline import RAGPipeline


CHUNKS = [
    {"chunk_id": "c1", "text": "Nicolas Flamel is the only known maker of the Philosopher's Stone.",
     "metadata": {"chunk_id": "c1", "source": "book.pdf", "page": 177, "section": "General"}},
    {"chunk_id": "c2", "text": "Harry plays Seeker for Gryffindor.",
     "metadata": {"chunk_id": "c2", "source": "book.pdf", "page": 121, "section": "General"}},
]


def tagged(chunks):
    from src.retrieval.retriever import tag_citations
    return tag_citations(chunks)


class FakeSparse:
    def search(self, query, top_k=5):
        return [{**c, "score": 3.0} for c in CHUNKS][:top_k]


class FakeRetriever:
    def __init__(self, top_score=0.9):
        self.top_score = top_score
        self.sparse = FakeSparse()

    def retrieve(self, query):
        chunks = tagged(CHUNKS)
        return {"query": query, "chunks": chunks, "context": "[1] ...\n\n[2] ...",
                "top_score": self.top_score}


class FakeGenerator:
    """Mimics Dev 3's GenerationEngine: refuses below the threshold."""
    def __init__(self, threshold, fail_with=None):
        self.threshold = threshold
        self.fail_with = fail_with
        self.calls = []

    def generate(self, retrieval):
        self.calls.append(retrieval)
        if self.fail_with:
            raise self.fail_with
        if retrieval["top_score"] < self.threshold:
            return {"status": "refused", "answer": "I couldn't find enough information.",
                    "citations": []}
        c = retrieval["chunks"][0]
        return {"status": "answered", "answer": "He made the Stone [1].",
                "citations": [{"id": "[1]", "document": c["document"], "page": c["page"],
                               "section": c["section"], "snippet": c["snippet"]}]}


def make_pipeline(top_score=0.9, fail_with=None, threshold=0.5):
    return RAGPipeline(FakeRetriever(top_score),
                       lambda t: FakeGenerator(t, fail_with), threshold=threshold)


@pytest.fixture
def client():
    yield TestClient(app)          # no "with": the lifespan (model loading) does not run
    app.state.pipeline = None
    app.state.startup_error = None


# ---- POST /ask -------------------------------------------------------------
def test_ask_returns_answer_with_citations(client):
    app.state.pipeline = make_pipeline()
    r = client.post("/ask", json={"question": "Who is Nicolas Flamel?"})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert "[1]" in body["answer"]
    assert body["citations"][0] == {
        "id": "[1]", "document": "book.pdf", "page": 177, "section": "General",
        "snippet": "Nicolas Flamel is the only known maker of the Philosopher's Stone.",
    }
    assert body["mode"] == "full"
    assert body["latency_ms"] >= 0


def test_ask_refuses_with_empty_citations(client):
    app.state.pipeline = make_pipeline(top_score=0.01)
    body = client.post("/ask", json={"question": "What is Polyjuice Potion?"}).json()

    assert body["status"] == "refused"
    assert body["citations"] == []


def test_ask_bm25_baseline_mode(client):
    app.state.pipeline = make_pipeline()
    body = client.post("/ask", json={"question": "Flamel?", "mode": "bm25_baseline"}).json()

    assert body["mode"] == "bm25_baseline"
    assert body["status"] == "answered"


@pytest.mark.parametrize("payload", [
    {},
    {"question": ""},
    {"question": "   "},
    {"question": "x" * 1001},
    {"question": "ok", "mode": "dense"},
])
def test_ask_rejects_invalid_input(client, payload):
    app.state.pipeline = make_pipeline()
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_reports_ollama_down(client):
    app.state.pipeline = make_pipeline(fail_with=requests.exceptions.ConnectionError())
    r = client.post("/ask", json={"question": "Who is Flamel?"})

    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]


def test_ask_unexpected_error_is_500(client):
    app.state.pipeline = make_pipeline(fail_with=ValueError("boom"))
    assert client.post("/ask", json={"question": "Who is Flamel?"}).status_code == 500


def test_ask_while_pipeline_missing(client):
    app.state.pipeline = None
    app.state.startup_error = "OSError: model not found"
    r = client.post("/ask", json={"question": "Who is Flamel?"})

    assert r.status_code == 503
    assert "model not found" in r.json()["detail"]


# ---- GET /health -------------------------------------------------------------
def test_health(client):
    app.state.pipeline = make_pipeline()
    assert client.get("/health").json() == {"status": "ok", "detail": None}

    app.state.pipeline = None
    app.state.startup_error = "boom"
    assert client.get("/health").json()["status"] == "error"


# ---- pipeline glue -------------------------------------------------------------
def test_bm25_baseline_skips_score_threshold():
    pipe = make_pipeline(threshold=0.99)
    res = pipe.ask("Flamel?", "bm25_baseline")

    assert res.status == "answered"          # threshold does not apply to BM25
    assert res.top_score is None
    assert [c["page"] for c in res.retrieved] == [177, 121]


def test_eval_mode_returns_llm_decision_without_threshold():
    pipe = make_pipeline(top_score=0.1, threshold=0.5)

    assert pipe.ask("q", "full").status == "refused"
    res = pipe.ask("q", "full", apply_threshold=False)
    assert res.llm_status == "answered"      # LLM decision is kept for the sweep
    assert res.top_score == 0.1


def test_unknown_mode():
    with pytest.raises(ValueError):
        make_pipeline().ask("q", "dense")
