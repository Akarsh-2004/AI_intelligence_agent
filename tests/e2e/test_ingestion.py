import pytest
from pathlib import Path
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.vector_store import VectorStore
from app.retrieval.embeddings import get_embeddings_model
from app.generation.generator import RAGGenerator

class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 384

def test_successful_document_ingestion(tmp_path, mocker):
    mocker.patch("app.retrieval.vector_store.get_embeddings_model", return_value=DummyEmbeddings())
    # Given a valid Markdown document exists
    doc_path = tmp_path / "test_doc.md"
    doc_path.write_text("# RAG Overview\nRetrieval-Augmented Generation is a technique...")
    
    # When the user submits the document for ingestion
    pipeline = IngestionPipeline()
    result = pipeline.ingest_document(str(doc_path))
    
    # Then the system should parse the document and split it into chunks
    # And generate embeddings for each chunk
    # And store the embeddings in the vector database
    # And preserve document metadata
    # And return the number of chunks successfully indexed
    
    assert result["status"] == "success"
    assert result["chunks_indexed"] > 0
    assert result["metadata"][0]["source"] == str(doc_path)
    
def test_reject_unsupported_documents(tmp_path, mocker):
    mocker.patch("app.retrieval.vector_store.get_embeddings_model", return_value=DummyEmbeddings())
    # Given a document has an unsupported file type
    doc_path = tmp_path / "test_doc.jpg"
    doc_path.write_text("fake image data")
    
    # When the user attempts to ingest it
    pipeline = IngestionPipeline()
    
    # Then the system should reject the document
    with pytest.raises(ValueError, match="Unsupported file type"):
        pipeline.ingest_document(str(doc_path))
