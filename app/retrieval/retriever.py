from typing import List, Any
from app.retrieval.vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore = None, k: int = 4):
        self.vector_store = vector_store or VectorStore()
        self.k = k

    def retrieve(self, query: str) -> List[Any]:
        # Return relevant chunks from vector store
        docs = self.vector_store.similarity_search(query, k=self.k)
        if not docs:
            # We can return empty and let the generator handle "insufficient context"
            return []
        return docs
