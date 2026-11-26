"""Ask questions against the local corpus with Ollama."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from os import devnull
from pathlib import Path

import pymupdf

from src.generation.generator import GenerationEngine
from src.generation.llm_client import LLMClient


@lru_cache(maxsize=None)
def _chapter_bookmarks(document: str) -> list[tuple[int, str]]:
    path = Path(__file__).resolve().parents[1] / "src" / "Data" / Path(document).name
    if path.suffix.lower() != ".pdf" or not path.is_file():
        return []

    with pymupdf.open(path) as pdf:
        return sorted(
            (page, title.strip())
            for _, title, page in pdf.get_toc()
            if page > 0 and title.strip().casefold().startswith("chapter ")
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Ask one question and exit")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be greater than 0 and at most 1")

    with open(devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        from src.retrieval.retriever import RetrievalEngine

        retriever = RetrievalEngine()
    generator = GenerationEngine(LLMClient.from_ollama(args.model), threshold=args.threshold)

    def ask(question):
        result = generator.generate(retriever.retrieve(question))
        print(result["answer"])
        for citation in result["citations"]:
            page = citation["page"]
            document = citation["document"]
            chapter = None
            if page is not None:
                chapter = next(
                    (title for start, title in reversed(_chapter_bookmarks(document)) if start <= page),
                    None,
                )
            details = [document]
            section = chapter or citation.get("section")
            if section:
                details.append(section)
            if page is not None:
                details.append(f"PDF page {page}")
            print(f'{citation["id"]} {", ".join(details)}')

    if args.question is not None:
        ask(args.question)
        return

    while True:
        try:
            question = input("\nAsk (or type exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() == "exit":
            return
        if question:
            ask(question)


if __name__ == "__main__":
    main()
