import argparse
from pathlib import Path

from src.ingestion.ingest_data import DATA_DIR, run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Index every supported file in the corpus folder.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    run_ingestion(args.data_dir)


if __name__ == "__main__":
    main()
