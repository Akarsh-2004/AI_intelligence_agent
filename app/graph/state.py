from typing import List, Dict, Any, TypedDict, Optional
import operator

class AgentState(TypedDict):
    query: str
    retrieved_documents: List[Any]
    answer: str
    critique: str
    evaluation: Dict[str, Any]
    retry_count: int
    max_retries: int
