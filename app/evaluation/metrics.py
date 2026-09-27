from typing import List, Dict, Any
import time

def calculate_recall_at_k(retrieved_docs: List[Any], expected_topics: List[str]) -> float:
    """Calculate recall by checking if expected topics appear in retrieved docs."""
    if not expected_topics:
        return 1.0
        
    combined_content = " ".join([doc.page_content.lower() for doc in retrieved_docs])
    found = sum(1 for topic in expected_topics if topic.lower() in combined_content)
    return found / len(expected_topics)

def calculate_mrr(retrieved_docs: List[Any], expected_topics: List[str]) -> float:
    """Calculate Mean Reciprocal Rank based on first appearance of any topic."""
    if not expected_topics:
        return 1.0
        
    for i, doc in enumerate(retrieved_docs):
        content = doc.page_content.lower()
        if any(topic.lower() in content for topic in expected_topics):
            return 1.0 / (i + 1)
    return 0.0

def calculate_answer_relevance(answer: str, expected_topics: List[str]) -> float:
    """Deterministic proxy for answer relevance (word overlap)."""
    if not expected_topics:
        return 1.0
        
    answer_lower = answer.lower()
    found = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
    return found / len(expected_topics)
