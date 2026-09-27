import pytest
from app.retrieval.retriever import Retriever
from app.generation.generator import RAGGenerator
from langchain_core.messages import AIMessage

from langchain_core.runnables import RunnableLambda

class MockLLM:
    def __call__(self, inputs):
        if "hybrid retrieval" in inputs.text.lower() or "hybrid retrieval" in inputs.to_string().lower():
            return AIMessage(content="Hybrid retrieval combines dense and sparse retrieval methods.")
        return AIMessage(content="I don't know.")

def test_retrieve_relevant_information(tmp_path, mocker):
    # Setup mock vector store
    mock_vector_store = mocker.MagicMock()
    mock_doc = mocker.MagicMock()
    mock_doc.page_content = "Hybrid retrieval is a combination of dense and sparse retrieval."
    mock_doc.metadata = {"source": "rag_guide.md"}
    mock_vector_store.similarity_search.return_value = [mock_doc]
    
    # Given the knowledge base contains information about RAG
    retriever = Retriever(vector_store=mock_vector_store)
    
    # When the user asks
    query = "What is hybrid retrieval?"
    results = retriever.retrieve(query)
    
    # Then the system should search and return relevant chunks
    assert len(results) == 1
    assert "hybrid retrieval" in results[0].page_content.lower()

def test_no_relevant_information(mocker):
    # Given the knowledge base contains no relevant information
    mock_vector_store = mocker.MagicMock()
    mock_vector_store.similarity_search.return_value = []
    
    retriever = Retriever(vector_store=mock_vector_store)
    llm = MockLLM()
    generator = RAGGenerator(llm=llm)
    
    # When the user asks an unrelated question
    query = "What is the capital of France?"
    results = retriever.retrieve(query)
    response = generator.generate(query, results)
    
    # Then the retriever should indicate sufficient context was not found
    assert len(results) == 0
    # And generator should not fabricate knowledge
    assert "sufficient evidence was not found" in response["answer"].lower()

def test_generate_grounded_answer(mocker):
    # Given relevant documents exist
    mock_doc = mocker.MagicMock()
    mock_doc.page_content = "Hybrid retrieval combines dense and sparse retrieval methods."
    mock_doc.metadata = {"source": "rag_guide.md"}
    
    llm = MockLLM()
    generator = RAGGenerator(llm=llm)
    
    # When the user asks a question
    query = "What is hybrid retrieval?"
    
    # Then the system should generate an answer based on retrieved context
    response = generator.generate(query, [mock_doc])
    
    # And return source citations/metadata
    assert "hybrid retrieval" in response["answer"].lower()
    assert len(response["sources"]) == 1
    assert response["sources"][0]["source"] == "rag_guide.md"
