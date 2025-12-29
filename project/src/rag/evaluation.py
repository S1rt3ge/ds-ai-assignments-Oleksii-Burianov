from typing import List, Dict, Any
from src.rag.models import Chunk


class RAGEvaluator:
    def context_precision(self, retrieved_chunks: List[Chunk], relevant_chunk_ids: List[str]) -> float:
        if not retrieved_chunks:
            return 0.0

        retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        relevant_ids = set(relevant_chunk_ids)

        precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids)
        return precision

    def context_recall(self, retrieved_chunks: List[Chunk], relevant_chunk_ids: List[str]) -> float:
        if not relevant_chunk_ids:
            return 0.0

        retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        relevant_ids = set(relevant_chunk_ids)

        recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
        return recall

    def side_by_side_compare(
        self, query: str, rag_response: str, non_rag_response: str, context_used: int = 0
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "rag_response": rag_response,
            "non_rag_response": non_rag_response,
            "context_chunks_used": context_used,
            "rag_length": len(rag_response.split()),
            "non_rag_length": len(non_rag_response.split()),
        }
