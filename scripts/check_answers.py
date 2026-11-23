"""Live regression checks against the Book 1 corpus; requires local Ollama.

Run separately from the mocked unit tests: python -m scripts.check_answers
This command reads the existing index and never runs ingestion.
"""
import argparse
import re
import time

from src.generation.generator import GenerationEngine
from src.generation.llm_client import LLMClient
from src.retrieval.retriever import RetrievalEngine


# Gold answers belong in evaluation, never in retrieval or generation logic.
CASES = [
    {"question": "Who is Harry's mother?", "terms": ["lily"],
     "forbidden": ["weasley", "petunia"], "page": 42},
    {"question": "What is Harry's wand made of?", "terms": ["holly", "phoenix", "feather"],
     "forbidden": ["yew", "willow", "beechwood"], "page": 67},
    {"question": "What is Polyjuice Potion?", "refused": True},
]


def check(case, result):
    if case.get("refused"):
        return result["status"] == "refused" and not result["citations"]
    words = set(re.findall(r"[a-z]+", result["answer"].lower()))
    return (
        result["status"] == "answered"
        and all(term in words for term in case["terms"])
        and not any(term in words for term in case["forbidden"])
        and any(citation["page"] == case["page"] for citation in result["citations"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    args = parser.parse_args()
    retriever = RetrievalEngine()
    generator = GenerationEngine(LLMClient(args.model), threshold=0.5)
    passed = 0
    for case in CASES:
        start = time.perf_counter()
        result = generator.generate(retriever.retrieve(case["question"]))
        ok = check(case, result)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {case['question']} ({time.perf_counter() - start:.1f}s)", flush=True)
        print(result["answer"], flush=True)
        print(f"Citation pages: {[c['page'] for c in result['citations']]}", flush=True)
    print(f"{passed}/{len(CASES)} live checks passed.")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
