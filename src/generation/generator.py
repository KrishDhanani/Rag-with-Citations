import json

from src.generation.llm_client import LLMClient
from src.generation.refusal import should_refuse
from src.generation.citations import normalize_citations, citation_ids as get_citation_ids


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

        # Normalize grouped citations such as [1, 2] before validating each ID.
        answer = normalize_citations(answer)
        citation_ids = get_citation_ids(answer)
        if not citation_ids or any(label not in sources for label in citation_ids):
            return _refusal()

        # Show the supporting quote, rather than an unrelated page opening.
        supporting_quotes = {}
        evidence = response.get("evidence", [])
        for item in (evidence if isinstance(evidence, list) else []):
            if not isinstance(item, dict):
                continue
            label, quote = item.get("citation_id"), item.get("quote")
            if not isinstance(label, str) or label not in sources or not isinstance(quote, str):
                continue
            quote = " ".join(quote.split())
            original = " ".join(sources[label].get("text", "").split())
            if quote and quote in original:
                supporting_quotes.setdefault(label, []).append(quote)

        citations = [
            {
                "id": label,
                "document": sources[label]["document"],
                "page": sources[label]["page"],
                "section": sources[label]["section"],
                "snippet": " … ".join(dict.fromkeys(supporting_quotes[label]))
                if label in supporting_quotes else sources[label]["snippet"],
            }
            for label in citation_ids
        ]
        return {"status": "answered", "answer": answer.strip(), "citations": citations}
