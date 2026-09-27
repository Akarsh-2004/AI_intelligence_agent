import os
from app.ingestion.loaders import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.retrieval.vector_store import VectorStore

class IngestionPipeline:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore(persist_directory=persist_directory)

    def ingest_document(self, file_path: str) -> dict:
        try:
            # Parse document
            documents = DocumentLoader.load(file_path)
            
            # Split into chunks
            chunks = self.chunker.chunk_documents(documents)
            
            # Store in vector database
            self.vector_store.add_documents(chunks)
            
            # Return status
            return {
                "status": "success",
                "chunks_indexed": len(chunks),
                "metadata": [chunk.metadata for chunk in chunks]
            }
            
        except ValueError as e:
            raise e
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
