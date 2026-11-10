from pathlib import Path
from parser import Parser
from chunker import Chunking
from vector_store import Vector_Database

DATA_DIR = Path("../../src/data")

def run_ingestion():
    if not DATA_DIR.exists():
        print(f"Directory: {DATA_DIR} does not exists.")
        return

    parser = Parser()
    chunker = Chunking()
    db_manager = Vector_Database()
    for file_path in DATA_DIR.iterdir():
        if file_path.is_file() and not file_path.name.startswith("."):
            try:
                print("-" * 20)

                parsed_text = parser.parse_document(file_path=file_path)
                print(f"Result length: {len(parsed_text)} characters")
                print('-' * 20)

                chunked_docs = chunker.chunk_documents(documents=parsed_text)
                print(f"Processed Chunks: {len(chunked_docs)} chunks ")
                print('-' * 20)

                vector_store = db_manager.store_data(chunked_docs)
                print(f"Successfully stored {vector_store._collection.count} chunks into Chroma!")

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    run_ingestion()