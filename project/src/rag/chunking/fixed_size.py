from typing import List
from src.rag.models import Document, Chunk
from src.rag.chunking.base import ChunkingStrategy


class FixedSizeChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        tokens = self._tokenize(document.content)
        chunks = []
        chunk_index = 0

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)

            chunk = Chunk(
                text=chunk_text,
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                metadata=document.metadata,
            )
            chunks.append(chunk)

            chunk_index += 1
            start = end - self.overlap

        return chunks

    def _tokenize(self, text: str) -> List[str]:
        return text.split()
