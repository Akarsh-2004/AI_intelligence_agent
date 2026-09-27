from typing import List, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class RetrieverInput(BaseModel):
    query: str = Field(description="The query to search the knowledge base for.")
    
class CalculateInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate (e.g. 2 + 2).")

@tool("retrieve_documents", args_schema=RetrieverInput)
def retrieve_documents(query: str) -> str:
    """Retrieve documents from the knowledge base."""
    # In a real setup, this would use the global retriever instance
    return f"Retrieved documents for {query}"

@tool("calculate", args_schema=CalculateInput)
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Very simple and safe eval
        allowed_names = {"__builtins__": None}
        return str(eval(expression, allowed_names, {}))
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool("search_knowledge_base", args_schema=RetrieverInput)
def search_knowledge_base(query: str) -> str:
    """Search the broader knowledge base for information."""
    return f"Searched knowledge base for {query}"

TOOLS = [retrieve_documents, calculate, search_knowledge_base]
