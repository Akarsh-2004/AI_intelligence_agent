"""
Diagnostic script: test each pipeline component independently.
"""
import os, sys
os.environ.setdefault("OLLAMA_MODEL", "gemma2:2b")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("STEP 1: Test Ollama connectivity")
print("=" * 60)
try:
    import requests
    r = requests.get("http://localhost:11434", timeout=5)
    print(f"  Ollama OK: {r.status_code} — {r.text[:80]}")
except Exception as e:
    print(f"  Ollama FAIL: {e}")

print("\n" + "=" * 60)
print("STEP 2: Test ChromaDB local client")
print("=" * 60)
try:
    import chromadb
    client = chromadb.PersistentClient(path="./data/chroma_db")
    cols = client.list_collections()
    print(f"  ChromaDB OK — collections: {[c.name for c in cols]}")
except Exception as e:
    print(f"  ChromaDB FAIL: {e}")

print("\n" + "=" * 60)
print("STEP 3: Test VectorStore + similarity search")
print("=" * 60)
try:
    from app.retrieval.vector_store import VectorStore
    vs = VectorStore()
    docs = vs.similarity_search("what is machine learning", k=2)
    print(f"  VectorStore OK — got {len(docs)} docs")
    for d in docs:
        print(f"    snippet: {d.page_content[:80]!r}")
except Exception as e:
    print(f"  VectorStore FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("STEP 4: Test LLM call")
print("=" * 60)
try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="gemma2:2b", base_url="http://localhost:11434", temperature=0)
    resp = llm.invoke("Say hello in one sentence.")
    print(f"  LLM OK: {resp.content[:120]}")
except Exception as e:
    print(f"  LLM FAIL: {e}")

print("\n" + "=" * 60)
print("STEP 5: End-to-end single query")
print("=" * 60)
try:
    from app.retrieval.retriever import Retriever
    from app.generation.generator import RAGGenerator
    from app.graph.workflow import create_workflow
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="gemma2:2b", base_url="http://localhost:11434", temperature=0)
    vs = VectorStore()
    retriever = Retriever(vs)
    generator = RAGGenerator(llm)
    workflow = create_workflow(retriever, generator, critic_llm=llm)

    result = workflow.invoke({"query": "What is machine learning?", "retry_count": 0, "max_retries": 1})
    print(f"  Workflow result keys: {list(result.keys())}")
    answer = result.get("answer", result.get("generation", "NO ANSWER"))
    print(f"  Answer: {str(answer)[:200]}")
except Exception as e:
    print(f"  E2E FAIL: {e}")
    import traceback; traceback.print_exc()
