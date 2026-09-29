import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.graph.workflow import create_workflow
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.generation.generator import RAGGenerator
from langchain_ollama import ChatOllama

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "model": OLLAMA_MODEL}

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# --- Ollama LLM configuration ---
# Set OLLAMA_MODEL in your .env (default: gemma3:1b)
# Set OLLAMA_BASE_URL in your .env if Ollama is not on localhost (default: http://localhost:11434)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

# Initialize the pipeline globally (in production this would be in a lifespan event or dependency)
db = VectorStore()
retriever = Retriever(db)
generator = RAGGenerator(llm)
workflow = create_workflow(retriever, generator, critic_llm=llm)

@router.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    try:
        # Run the LangGraph workflow
        result = workflow.invoke({
            "query": request.query,
            "retry_count": 0,
            "max_retries": 3
        })

        # The generate_node stores the answer in "answer"
        answer = result.get("answer", result.get("generation", "No answer produced."))

        # Handle dict answers
        if isinstance(answer, dict):
            answer = answer.get("answer", str(answer))

        # Collect sources from retrieved documents in state
        sources = []
        docs = result.get("retrieved_documents", result.get("context", []))
        if docs:
            sources = [doc.metadata for doc in docs if hasattr(doc, "metadata")]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
