from pathlib import Path
from typing import Tuple
import pypdf
from src.rag.models import Document, DocumentMetadata


class DocumentProcessor:
    def __init__(self):
        self.supported_types = ["pdf", "txt", "md"]

    def ingest(self, file_path: str) -> Document:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_type = path.suffix.lower().lstrip(".")

        if doc_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {doc_type}. Supported: {self.supported_types}")

        if doc_type == "pdf":
            content, metadata = self._parse_pdf(path)
        elif doc_type == "txt":
            content, metadata = self._parse_txt(path)
        elif doc_type == "md":
            content, metadata = self._parse_markdown(path)
        else:
            raise ValueError(f"Unexpected document type: {doc_type}")

        return Document(content=content, metadata=metadata)

    def _parse_pdf(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            page_count = len(reader.pages)

            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())

            content = "\n\n".join(text_parts)

            metadata_dict = reader.metadata or {}
            author = metadata_dict.get("/Author", None)

            metadata = DocumentMetadata(
                filename=file_path.name,
                doc_type="pdf",
                page_count=page_count,
                author=author,
            )

        return content, metadata

    def _parse_txt(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = DocumentMetadata(
            filename=file_path.name,
            doc_type="txt",
        )

        return content, metadata

    def _parse_markdown(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = DocumentMetadata(
            filename=file_path.name,
            doc_type="md",
        )

        return content, metadata
