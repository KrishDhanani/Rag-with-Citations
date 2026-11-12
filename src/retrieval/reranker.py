import math

from sentence_transformers import CrossEncoder

from src import config


class Reranker:
    """
    Second-stage ranking with a cross-encoder. Unlike the bi-encoder used for
    dense search, it reads the query and the passage together, which is more
    accurate but too slow for the whole corpus. So it only scores the fused
    candidates.
    """

    def __init__(self, model=None):
        self.model = model or CrossEncoder(config.RERANKER_MODEL)

    def rerank(self, query, candidates, top_k=config.FINAL_TOP_K):
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        logits = self.model.predict(pairs)

        reranked = []
        for cand, logit in zip(candidates, logits):
            logit = float(logit)
            reranked.append({
                **cand,
                "rerank_score": logit,
                # sigmoid maps the raw logit to 0..1, easier to threshold
                # in the refusal logic than an unbounded value
                "confidence": 1 / (1 + math.exp(-logit)),
            })

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
