from langgraph.graph import StateGraph, END, START
from app.graph.state import AgentState
from app.graph.nodes import GraphNodes
from app.graph.edges import route_critique
from app.retrieval.retriever import Retriever
from app.generation.generator import RAGGenerator

def create_workflow(retriever: Retriever, generator: RAGGenerator, critic_llm=None):
    nodes = GraphNodes(retriever, generator, critic_llm)
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retriever", nodes.retrieve_node)
    workflow.add_node("generator", nodes.generate_node)
    workflow.add_node("critic", nodes.critic_node)
    workflow.add_node("evaluator", nodes.evaluate_node)
    
    # Define edges
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "critic")
    
    # Conditional edge from critic
    workflow.add_conditional_edges(
        "critic",
        route_critique,
        {
            "retriever": "retriever",
            "evaluator": "evaluator",
            "end": END
        }
    )
    
    workflow.add_edge("evaluator", END)
    
    return workflow.compile()
