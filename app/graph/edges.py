from typing import Literal
from app.graph.state import AgentState

def route_critique(state: AgentState) -> Literal["retriever", "evaluator", "end"]:
    critique = state.get("critique")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    if critique == "accepted":
        return "evaluator"
    
    if retry_count >= max_retries:
        return "end"
        
    return "retriever"
