from langchain_community.vectorstores import Chroma
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

class Vector_Database:
    def __init__(self):
        self.DIR = Path("../../src/Vectorstore")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def store_data(self, documents: List[Document]) -> Chroma:
        """Embeds and stores chunked documents into a local Chroma vector database."""
        vector_store = Chroma.from_documents(
            documents=documents,             
            embedding=self.embeddings,       
            persist_directory=str(self.DIR), 
            collection_name='rag_collection'
        )
        return vector_store