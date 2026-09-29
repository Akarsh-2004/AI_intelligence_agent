"""
run_pipeline.py – Full end-to-end pipeline demo with Ollama.

Usage:
    python run_pipeline.py

Requirements:
    - Ollama running locally: `ollama serve`
    - Model pulled: `ollama pull llama3`
    - Chroma DB already populated (run run_ingestion.py first if needed)
"""
import sys
from dotenv import load_dotenv
load_dotenv()

from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.generation.generator import RAGGenerator
from app.graph.workflow import create_workflow
from langchain_ollama import ChatOllama
import os

# ── LLM ──────────────────────────────────────────────────────────────────────
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"\n{'='*60}")
print(f"  AI Intelligence Agent – Full Pipeline Demo")
print(f"  LLM : {OLLAMA_MODEL}  @  {OLLAMA_BASE_URL}")
print(f"{'='*60}\n")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

# ── Pipeline components ───────────────────────────────────────────────────────
db        = VectorStore()
retriever = Retriever(db)
generator = RAGGenerator(llm)
workflow  = create_workflow(retriever, generator, critic_llm=llm)

# ── Use-case queries ──────────────────────────────────────────────────────────
USE_CASES = [
    # ① Core ML concepts
    ("What is machine learning and how does it differ from traditional programming?",
     "ML Fundamentals"),

    # ② Deep learning
    ("Explain activation functions and why they are important in neural networks.",
     "Deep Learning"),

    # ③ MLOps & deployment
    ("What is MLOps and what are its key practices for deploying ML models in production?",
     "MLOps"),

    # ④ Clustering / unsupervised learning
    ("What clustering algorithms are available in scikit-learn and when should I use each?",
     "Unsupervised Learning"),

    # ⑤ Career / roles
    ("What skills does an AI Engineer need compared to a Data Scientist?",
     "AI Career Paths"),

    # ⑥ Computer vision
    ("What is computer vision and what are its real-world applications?",
     "Computer Vision"),

    # ⑦ Regularisation
    ("How does regularization prevent overfitting in machine learning models?",
     "Model Regularization"),
]

DIVIDER = "-" * 60

def run_query(question: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  USE CASE: {label}")
    print(DIVIDER)
    print(f"  Q: {question}")
    print(DIVIDER)

    try:
        result = workflow.invoke({
            "query":       question,
            "retry_count": 0,
            "max_retries": 3,
        })

        answer  = result.get("answer", result.get("generation", "No answer produced."))
        context = result.get("retrieved_documents", [])
        sources = list({doc.metadata.get("source", "unknown") for doc in context})

        if isinstance(answer, dict):
            answer = answer.get("answer", str(answer))

        print(f"\n  A: {answer.strip()}")
        if sources:
            print(f"\n  Sources ({len(sources)}):")
            for s in sources[:3]:
                print(f"     * {s}")
        critique = result.get("critique", "n/a")
        print(f"\n  Critic verdict: {critique}")

    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)

    print()

# ── Run all use cases ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    for question, label in USE_CASES:
        run_query(question, label)

    print(f"\n{'='*60}")
    print("  All use-case queries completed.")
    print(f"{'='*60}\n")
