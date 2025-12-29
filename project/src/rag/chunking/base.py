from abc import ABC, abstractmethod
from typing import List
from src.rag.models import Document, Chunk


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass
