from src.rag.chunking.base import ChunkingStrategy
from src.rag.chunking.fixed_size import FixedSizeChunker
from src.rag.chunking.recursive import RecursiveChunker


def get_chunking_strategy(strategy_name: str) -> ChunkingStrategy:
    if strategy_name == "fixed":
        return FixedSizeChunker(chunk_size=500, overlap=50)
    elif strategy_name == "recursive":
        return RecursiveChunker()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")


__all__ = [
    "ChunkingStrategy",
    "FixedSizeChunker",
    "RecursiveChunker",
    "get_chunking_strategy",
]
