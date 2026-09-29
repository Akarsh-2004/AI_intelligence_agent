from app.graph.state import AgentState
from app.retrieval.retriever import Retriever
from app.generation.generator import RAGGenerator
from app.agents.schemas import CritiqueResult

class GraphNodes:
    def __init__(self, retriever: Retriever, generator: RAGGenerator, critic_llm=None):
        self.retriever = retriever
        self.generator = generator
        # Bind structured output to the critic LLM if provided
        self.critic_llm = critic_llm.with_structured_output(CritiqueResult) if critic_llm else None

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
        answer = state.get("answer", "")
        docs = state.get("retrieved_documents", [])
        retry_count = state.get("retry_count", 0) + 1

        # Fast-path: nothing to critique
        if not docs or not answer:
            return {"critique": "rejected", "retry_count": retry_count}

        # Use real Ollama-powered structured critique if available
        if self.critic_llm:
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = (
                f"You are a strict critic. Evaluate whether the following answer is fully supported "
                f"by the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"Answer:\n{answer}"
            )
            try:
                result: CritiqueResult = self.critic_llm.invoke(prompt)
                verdict = "accepted" if result.supported and result.confidence >= 0.6 else "rejected"
                return {"critique": verdict, "retry_count": retry_count}
            except Exception:
                # Fallback to heuristic if structured output fails
                pass

        # Heuristic fallback
        if "sufficient evidence was not found" in answer.lower():
            return {"critique": "rejected", "retry_count": retry_count}
        return {"critique": "accepted", "retry_count": retry_count}

    def evaluate_node(self, state: AgentState) -> AgentState:
        return {"evaluation": {"status": "completed"}}
