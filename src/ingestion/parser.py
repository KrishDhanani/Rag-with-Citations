from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Any
import pymupdf
import re

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class Parser:
    """Parses various file formats into standard Document objects with rich metadata."""
    
    def __init__(self):
        self.PARSER_REGISTRY: Dict[str, Callable[[Path], List[Document]]] = {
            ".pdf": self.parse_pdf,
            ".md": self.parse_markdown,
            ".markdown": self.parse_markdown,
            ".txt": self.parse_txt,
        }

    def _extract_section_heading(self, text: str) -> str:
        """Helper to find the first heading or prominent section title in the text."""
        match = re.search(r'^(#+\s+(.+)|[A-Z\s]{4,})$', text, re.MULTILINE)
        if match:
            return match.group(2) or match.group(1).strip()
        return "General"

    def parse_pdf(self, file_path: Path) -> List[Document]:
        docs = []
        with pymupdf.open(file_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text.strip():
                    section_title = self._extract_section_heading(text)
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "file_type": "pdf",
                                "page": page_num,
                                "section": section_title
                            }
                        )
                    )
        # print(docs[5].metadata)
        return docs

    def parse_markdown(self, file_path: Path) -> List[Document]:
        content = file_path.read_text(encoding="utf-8")
        return [
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "file_type": "md",
                    "page": 1,
                    "section": self._extract_section_heading(content)
                }
            )
        ]

    def parse_txt(self, file_path: Path) -> List[Document]:
        content = file_path.read_text(encoding="utf-8")
        return [
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "file_type": "txt",
                    "page": 1,
                    "section": "General"
                }
            )
        ]

    def parse_document(self, file_path: str | Path) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.PARSER_REGISTRY:
            raise ValueError(f"Unsupported format '{ext}'")

        return self.PARSER_REGISTRY[ext](path)