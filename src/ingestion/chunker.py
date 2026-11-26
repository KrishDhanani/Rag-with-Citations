"""Group complete paragraphs while retaining page-local citations."""
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunking:
    def __init__(self, chunk_size=750, chunk_overlap=150):
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=[". ", "! ", "? ", "; ", " ", ""],
        )

    def chunk_documents(self, documents):
        chunks = []
        for doc in documents:
            paragraphs = [re.sub(r"\s+", " ", p).strip()
                          for p in re.split(r"\n\s*\n", doc.page_content) if p.strip()]
            groups, current = [], []
            for paragraph in paragraphs:
                if len(paragraph) > self.chunk_size:
                    if current:
                        groups.append("\n\n".join(current))
                        current = []
                    groups.extend(self.fallback.split_text(paragraph))
                    continue
                if current and len("\n\n".join(current + [paragraph])) > self.chunk_size:
                    groups.append("\n\n".join(current))
                    overlap = []
                    for previous in reversed(current):
                        if len("\n\n".join([previous] + overlap)) > self.chunk_overlap:
                            break
                        overlap.insert(0, previous)
                    current = overlap
                    while current and len("\n\n".join(current + [paragraph])) > self.chunk_size:
                        current.pop(0)
                current.append(paragraph)
            if current:
                groups.append("\n\n".join(current))
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 1)
            for index, text in enumerate(groups, 1):
                chunks.append(Document(page_content=text, metadata={
                    **doc.metadata,
                    "chunk_id": f"{source}_p{page}_c{index}",
                    "chunk_index": index,
                    "char_count": len(text),
                    "chunking_version": "paragraph-v1",
                }))
        return chunks
