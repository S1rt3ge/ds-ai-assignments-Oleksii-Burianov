from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4


class DocumentMetadata(BaseModel):
    filename: str
    doc_type: str
    upload_date: datetime = Field(default_factory=datetime.now)
    page_count: Optional[int] = None
    author: Optional[str] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    doc_id: str
    chunk_index: int
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

    def flatten_metadata(self) -> Dict[str, Any]:
        return {
            "filename": self.metadata.filename,
            "doc_type": self.metadata.doc_type,
            "upload_date": self.metadata.upload_date.isoformat(),
            "page_count": self.metadata.page_count or 0,
            "author": self.metadata.author or "",
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
        }


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: DocumentMetadata
    chunks: List[Chunk] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int


class RAGContext(BaseModel):
    query: str
    results: List[RetrievalResult]
    top_k: int
    retrieval_time_ms: float


class RAGResponse(BaseModel):
    query: str
    answer: str
    context: RAGContext
    citations: List[str]
    rag_enabled: bool
