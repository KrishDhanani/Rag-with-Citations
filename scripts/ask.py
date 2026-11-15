"""Ask questions against the local corpus with Ollama."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from os import devnull

from src.generation.generator import GenerationEngine
from src.generation.llm_client import LLMClient


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
            location = f", PDF page {page}" if page is not None else ""
            print(f'{citation["id"]} {citation["document"]}{location}')

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
