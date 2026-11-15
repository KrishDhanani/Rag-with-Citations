import math


def should_refuse(context: str, top_score: float | None, *, threshold: float) -> bool:
    """Decide whether retrieval should stop the request before the LLM call."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1")

    return (
        not context.strip()
        or top_score is None
        or not math.isfinite(top_score)
        or not threshold <= top_score <= 1
    )
