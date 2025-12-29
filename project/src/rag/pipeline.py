from typing import List
from pathlib import Path
from src.rag.models import RAGContext
from src.rag.vector_store import ChromaDBStore
from src.rag.embeddings import EmbeddingModel
from src.rag.retriever import Retriever
from src.rag.document_processor import DocumentProcessor
from src.rag.chunking import get_chunking_strategy


class RAGPipeline:
    def __init__(
        self,
        vector_store: ChromaDBStore = None,
        embedding_model: EmbeddingModel = None,
        retriever: Retriever = None,
    ):
        self.vector_store = vector_store or ChromaDBStore()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.retriever = retriever or Retriever(self.vector_store, self.embedding_model)
        self.document_processor = DocumentProcessor()

    def index_document(self, file_path: str, chunking_strategy: str = "fixed"):
        document = self.document_processor.ingest(file_path)

        chunker = get_chunking_strategy(chunking_strategy)
        chunks = chunker.chunk(document)

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self.vector_store.add_chunks(chunks, embeddings)

        return len(chunks)

    def query(self, query: str, top_k: int = 5) -> RAGContext:
        return self.retriever.retrieve(query, top_k)

    def generate_rag_prompt(self, query: str, context: RAGContext) -> str:
        context_str = "\n\n".join(
            [
                f"[Source {r.rank}: {r.chunk.metadata.filename}, chunk {r.chunk.chunk_index}]\n{r.chunk.text}"
                for r in context.results
            ]
        )

        return f"""Use the following context to answer the question. Include inline citations.

Context:
{context_str}

Question: {query}

Answer with citations:"""

    def extract_citations(self, context: RAGContext) -> List[str]:
        return [
            f"Source: {r.chunk.metadata.filename}, chunk {r.chunk.chunk_index}" for r in context.results
        ]

    def get_indexed_count(self) -> int:
        return self.vector_store.count()

    def clear_index(self):
        self.vector_store.clear()
