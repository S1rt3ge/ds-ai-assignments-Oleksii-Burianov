import chromadb
from typing import List, Dict, Any
from src.rag.models import Chunk


class ChromaDBStore:
    def __init__(self, persist_directory: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        if not chunks:
            return

        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.flatten_metadata() for chunk in chunks],
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
