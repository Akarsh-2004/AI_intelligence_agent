import pytest
from app.graph.workflow import create_workflow

class MockRetriever:
    def __init__(self, succeed_on_second=False):
        self.call_count = 0
        self.succeed_on_second = succeed_on_second
        
    def retrieve(self, query):
        self.call_count += 1
        if self.succeed_on_second and self.call_count >= 2:
            mock_doc = type('obj', (object,), {'page_content': 'sufficient evidence', 'metadata': {}})
            return [mock_doc]
        if not self.succeed_on_second:
            mock_doc = type('obj', (object,), {'page_content': 'sufficient evidence', 'metadata': {}})
            return [mock_doc]
        return []

class MockGenerator:
    def generate(self, query, docs):
        if not docs:
            return {"answer": "sufficient evidence was not found", "sources": []}
        return {"answer": "Here is a good answer.", "sources": []}

def test_workflow_accepts_good_answer():
    # Given the retriever returns sufficient evidence
    retriever = MockRetriever(succeed_on_second=False)
    generator = MockGenerator()
    app = create_workflow(retriever, generator)
    
    # When the graph is executed
    result = app.invoke({"query": "test query", "retry_count": 0, "max_retries": 3})
    
    # Then the workflow should continue to evaluation
    assert result["critique"] == "accepted"
    assert "evaluation" in result
    assert result["retry_count"] == 1
    assert retriever.call_count == 1

def test_workflow_rejects_and_retries():
    # Given the retrieved context is insufficient initially
    retriever = MockRetriever(succeed_on_second=True)
    generator = MockGenerator()
    app = create_workflow(retriever, generator)
    
    # When the graph is executed
    result = app.invoke({"query": "test query", "retry_count": 0, "max_retries": 3})
    
    # Then the LangGraph workflow should route back to retrieval
    assert result["critique"] == "accepted" # Accepted on the second try
    assert result["retry_count"] == 2
    assert retriever.call_count == 2
    
def test_workflow_prevents_infinite_retries():
    class FailingRetriever:
        def retrieve(self, query):
            return []
            
    retriever = FailingRetriever()
    generator = MockGenerator()
    app = create_workflow(retriever, generator)
    
    result = app.invoke({"query": "test query", "retry_count": 0, "max_retries": 2})
    
    assert result["critique"] == "rejected"
    assert result["retry_count"] == 2
    assert "evaluation" not in result
