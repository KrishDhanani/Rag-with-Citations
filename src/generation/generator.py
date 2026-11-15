import json
import re

from src.generation.llm_client import LLMClient
from src.generation.refusal import should_refuse


def _refusal() -> dict:
    return {
        "status": "refused",
        "answer": "I couldn't find enough information in the provided sources to answer that question.",
        "citations": [],
    }


class GenerationEngine:
    def __init__(self, client: LLMClient, *, threshold: float):
        self.client = client
        self.threshold = threshold

    def generate(self, retrieval_result: dict) -> dict:
        """Convert a RetrievalEngine result into an answer or refusal."""
        context = retrieval_result["context"]
        if should_refuse(context, retrieval_result["top_score"], threshold=self.threshold):
            return _refusal()

        sources = {chunk["citation_id"]: chunk for chunk in retrieval_result["chunks"]}
        if not sources:
            return _refusal()

        raw_response = self.client.complete(retrieval_result["query"], context)
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError:
            return _refusal()

        if not isinstance(response, dict) or response.get("status") != "answered":
            return _refusal()
        answer = response.get("answer")
        if not isinstance(answer, str):
            return _refusal()

        citation_ids = list(dict.fromkeys(re.findall(r"\[[0-9]+\]", answer)))
        if not citation_ids or any(label not in sources for label in citation_ids):
            return _refusal()

        citations = [
            {
                "id": label,
                "document": sources[label]["document"],
                "page": sources[label]["page"],
                "section": sources[label]["section"],
                "snippet": sources[label]["snippet"],
            }
            for label in citation_ids
        ]
        return {"status": "answered", "answer": answer.strip(), "citations": citations}
