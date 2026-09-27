from app.graph.state import AgentState
from app.retrieval.retriever import Retriever
from app.generation.generator import RAGGenerator

class GraphNodes:
    def __init__(self, retriever: Retriever, generator: RAGGenerator, critic_llm=None):
        self.retriever = retriever
        self.generator = generator
        self.critic_llm = critic_llm
        
    def retrieve_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        docs = self.retriever.retrieve(query)
        return {"retrieved_documents": docs}
        
    def generate_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        docs = state.get("retrieved_documents", [])
        
        response = self.generator.generate(query, docs)
        return {"answer": response["answer"]}
        
    def critic_node(self, state: AgentState) -> AgentState:
        # Mock critic for now, we will implement actual critic later
        answer = state.get("answer", "")
        docs = state.get("retrieved_documents", [])
        
        if not docs or "sufficient evidence was not found" in answer.lower():
            # Insufficient context, reject
            return {"critique": "rejected", "retry_count": state.get("retry_count", 0) + 1}
            
        # Accept by default if there's an answer and docs
        return {"critique": "accepted", "retry_count": state.get("retry_count", 0) + 1}
        
    def evaluate_node(self, state: AgentState) -> AgentState:
        return {"evaluation": {"status": "completed"}}
