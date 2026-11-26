from pathlib import Path

from src import config
from src.ingestion.parser import Parser
from src.ingestion.chunker import Chunking
from src.ingestion.vector_store import Vector_Database
from src.ingestion.contextualizer import Contextualizer

DATA_DIR = config.BASE_DIR / "Data"


def run_ingestion(data_dir: Path = DATA_DIR, *, contextual=True,
                  context_model=config.CONTEXT_MODEL):
    files = sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}
    )
    if not files:
        raise ValueError(f"No supported documents found in {data_dir}")

    parser = Parser()
    chunker = Chunking()
    database = Vector_Database()
    for path in files:
        documents = parser.parse_document(path)
        chunks = chunker.chunk_documents(documents)
        if not chunks:
            raise ValueError(f"No text extracted from {path.name}; check whether it needs OCR")
        if contextual:
            Contextualizer(context_model, config.BASE_DIR.parent / ".cache" / "contextual").enrich(documents, chunks)
        database.store_data(chunks)
        print(f"Indexed {path.name}: {len(documents)} pages/documents, {len(chunks)} chunks")
    print(f"Complete: {len(files)} files indexed.")


if __name__ == "__main__":
    run_ingestion()
