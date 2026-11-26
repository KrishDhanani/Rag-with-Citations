from hashlib import sha256
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src import config
from src.ingestion.contextualizer import embedding_text


class Vector_Database:
    def __init__(self):
        self.DIR = config.VECTOR_DB_DIR
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

    def store_data(self, documents: List[Document]) -> Chroma:
        """Upsert a complete source, then remove its obsolete and legacy records.

        Run one ingestion process at a time. If a batch fails, rerun the command
        to finish that source before restarting the question-answering service.
        """
        vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.DIR),
            collection_name=config.COLLECTION_NAME,
        )
        sources = sorted({doc.metadata["source"] for doc in documents})
        for source in sources:
            source_docs = [doc for doc in documents if doc.metadata["source"] == source]
            previous = vector_store.get(where={"source": source})["ids"]
            model = self.embeddings._client
            indexed_texts = [embedding_text(doc.page_content, doc.metadata,
                                            model.tokenizer, model.max_seq_length)
                             for doc in source_docs]
            ids = [
                sha256(f"{source}\0{doc.metadata['chunk_id']}\0{text}".encode()).hexdigest()
                for doc, text in zip(source_docs, indexed_texts)
            ]
            for start in range(0, len(source_docs), 128):
                batch = source_docs[start:start + 128]
                embeddings = self.embeddings.embed_documents(indexed_texts[start:start + 128])
                # Store original passages, but embed context-enriched text.
                vector_store._collection.upsert(
                    ids=ids[start:start + 128], embeddings=embeddings,
                    documents=[doc.page_content for doc in batch],
                    metadatas=[doc.metadata for doc in batch],
                )
            stale = sorted(set(previous) - set(ids))
            for start in range(0, len(stale), 128):
                vector_store.delete(ids=stale[start:start + 128])
        return vector_store
