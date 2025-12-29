from src.rag.models import RetrievalResult, RAGContext, Chunk, DocumentMetadata
from src.rag.vector_store import ChromaDBStore
from src.rag.embeddings import EmbeddingModel
from datetime import datetime
import time


class Retriever:
    def __init__(self, vector_store: ChromaDBStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> RAGContext:
        start = time.time()

        query_embedding = self.embedding_model.embed_text(query)

        raw_results = self.vector_store.search(query_embedding, top_k)

        results = []
        for i in range(len(raw_results["ids"][0])):
            chunk_id = raw_results["ids"][0][i]
            distance = raw_results["distances"][0][i]
            metadata = raw_results["metadatas"][0][i]
            document_text = raw_results["documents"][0][i]

            chunk = self._reconstruct_chunk(chunk_id, document_text, metadata)
            score = 1 - distance

            results.append(RetrievalResult(chunk=chunk, score=score, rank=i + 1))

        elapsed = (time.time() - start) * 1000

        return RAGContext(query=query, results=results, top_k=top_k, retrieval_time_ms=elapsed)

    def _reconstruct_chunk(self, chunk_id: str, text: str, metadata: dict) -> Chunk:
        doc_metadata = DocumentMetadata(
            filename=metadata.get("filename", ""),
            doc_type=metadata.get("doc_type", ""),
            upload_date=datetime.fromisoformat(metadata.get("upload_date", datetime.now().isoformat())),
            page_count=metadata.get("page_count", 0) or None,
            author=metadata.get("author", "") or None,
        )

        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            doc_id=metadata.get("doc_id", ""),
            chunk_index=metadata.get("chunk_index", 0),
            metadata=doc_metadata,
        )

        return chunk
