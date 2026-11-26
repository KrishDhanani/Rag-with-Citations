import argparse
from pathlib import Path

from src.ingestion.ingest_data import DATA_DIR, run_ingestion
from src import config


def main():
    parser = argparse.ArgumentParser(description="Index every supported file in the corpus folder.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--context-model", default=config.CONTEXT_MODEL)
    parser.add_argument("--no-context", action="store_true", help="Paragraph chunking without generated context")
    args = parser.parse_args()
    run_ingestion(args.data_dir, contextual=not args.no_context, context_model=args.context_model)


if __name__ == "__main__":
    main()
