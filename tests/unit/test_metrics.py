from app.evaluation.metrics import calculate_recall_at_k, calculate_mrr, calculate_answer_relevance

class MockDoc:
    def __init__(self, content):
        self.page_content = content

def test_metrics():
    docs = [
        MockDoc("This talks about dense retrieval."),
        MockDoc("This talks about sparse retrieval and combination.")
    ]
    topics = ["dense retrieval", "sparse retrieval", "combination"]
    
    # Recall should be 1.0 (all topics found in docs)
    assert calculate_recall_at_k(docs, topics) == 1.0
    
    # MRR should be 1.0 (first doc contains "dense retrieval")
    assert calculate_mrr(docs, topics) == 1.0
    
    answer = "Hybrid retrieval uses dense retrieval and sparse retrieval."
    # Relevance should be 2/3 = 0.666...
    assert abs(calculate_answer_relevance(answer, topics) - 0.666) < 0.01
