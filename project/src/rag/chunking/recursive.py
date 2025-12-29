from typing import List
from src.rag.models import Document, Chunk
from src.rag.chunking.base import ChunkingStrategy


class RecursiveChunker(ChunkingStrategy):
    def __init__(self, separators: List[str] = None, max_chunk_size: int = 500):
        self.separators = separators or ["\n\n", "\n", ". "]
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> List[Chunk]:
        splits = self._recursive_split(document.content, self.separators)
        merged_chunks = self._merge_small_chunks(splits)

        chunks = []
        for idx, text in enumerate(merged_chunks):
            chunk = Chunk(
                text=text.strip(),
                doc_id=document.doc_id,
                chunk_index=idx,
                metadata=document.metadata,
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator)

        result = []
        for split in splits:
            if self._count_tokens(split) <= self.max_chunk_size:
                result.append(split)
            else:
                if remaining_separators:
                    result.extend(self._recursive_split(split, remaining_separators))
                else:
                    words = split.split()
                    for i in range(0, len(words), self.max_chunk_size):
                        result.append(" ".join(words[i : i + self.max_chunk_size]))

        return result

    def _merge_small_chunks(self, chunks: List[str], min_size: int = 100) -> List[str]:
        if not chunks:
            return []

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            if self._count_tokens(current_chunk) < min_size:
                current_chunk = current_chunk + " " + next_chunk
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk

        merged.append(current_chunk)
        return merged

    def _count_tokens(self, text: str) -> int:
        return len(text.split())
