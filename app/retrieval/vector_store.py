from typing import List, Any
from langchain_chroma import Chroma
from app.retrieval.embeddings import get_embeddings_model

class VectorStore:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.embeddings = get_embeddings_model()
        self.persist_directory = persist_directory
        self.db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
    def add_documents(self, documents: List[Any]) -> List[str]:
        return self.db.add_documents(documents)
        
    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        return self.db.similarity_search(query, k=k)
