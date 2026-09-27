import pytest
import time
from app.evaluation.metrics import calculate_recall_at_k, calculate_mrr, calculate_answer_relevance

def test_evaluate_complete_rag_system(mocker):
    # Given an evaluation dataset
    dataset = [
        {
            "id": "rag_001",
            "question": "What is hybrid retrieval?",
            "expected_topics": ["dense retrieval", "sparse retrieval", "combination"]
        },
        {
            "id": "rag_002",
            "question": "What is RAG?",
            "expected_topics": ["retrieval", "generation", "llm"]
        }
    ]
    
    # Mock workflow invoke
    mock_workflow = mocker.MagicMock()
    
    def side_effect(inputs):
        time.sleep(0.1) # Simulate latency
        mock_doc = mocker.MagicMock()
        mock_doc.page_content = " ".join(dataset[0]["expected_topics"] if "hybrid" in inputs["query"] else dataset[1]["expected_topics"])
        
        return {
            "retrieved_documents": [mock_doc],
            "answer": "A grounded answer containing " + mock_doc.page_content,
            "retry_count": 1
        }
        
    mock_workflow.invoke.side_effect = side_effect
    
    # When the evaluation harness is executed
    results = []
    start_time = time.time()
    
    for case in dataset:
        case_start = time.time()
        res = mock_workflow.invoke({"query": case["question"]})
        latency = time.time() - case_start
        
        recall = calculate_recall_at_k(res["retrieved_documents"], case["expected_topics"])
        mrr = calculate_mrr(res["retrieved_documents"], case["expected_topics"])
        relevance = calculate_answer_relevance(res["answer"], case["expected_topics"])
        
        results.append({
            "id": case["id"],
            "recall": recall,
            "mrr": mrr,
            "relevance": relevance,
            "latency": latency,
            "retries": res["retry_count"]
        })
        
    total_latency = time.time() - start_time
    
    # Then metrics should be calculated
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    
    assert avg_recall == 1.0
    assert avg_mrr == 1.0
    assert avg_relevance == 1.0
    assert total_latency > 0.2
