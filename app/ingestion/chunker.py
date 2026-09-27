from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        # Split documents and ensure metadata is preserved
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk ID to metadata
        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = i
                
        return chunks
