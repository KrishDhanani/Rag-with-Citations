import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class Chunking:
    """
        Splits documents into smart chunks using LangChain's RecursiveCharacterTextSplitter,
        while dynamically tracking and updating 'section' metadata as new headings appear.
    """

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        
        chunked_docs = []
        
        for doc in documents:
            content = doc.page_content
            source_name = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", 1)
            
            current_section = doc.metadata.get("section", "General")
            
            splits = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[doc.metadata]
            )
            
            for chunk_idx, split_doc in enumerate(splits, start=1):
                chunk_text = split_doc.page_content
                
                header_match = re.search(r'^(#+\s+(.+)|[A-Z\s]{4,})$', chunk_text, re.MULTILINE)    # Finding new header if any
                if header_match:
                    current_section = header_match.group(2) or header_match.group(1).strip()
                
                safe_source_name = source_name.rsplit('.', 1)[0].replace(" ", "_")  # Unique vhunk id creation
                chunk_id = f"{safe_source_name}_p{page_num}_c{chunk_idx}"
                
                updated_metadata = {
                    **split_doc.metadata,
                    "section": current_section,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "char_count": len(chunk_text)
                }
                
                chunked_docs.append(
                    Document(page_content=chunk_text, metadata=updated_metadata)
                )
                
        return chunked_docs