# AI Engineering Intelligence Platform

An AI-centric knowledge and evaluation platform demonstrating production-oriented RAG and Agent systems.

## Architecture

This project is built using a layered architecture to ensure modularity, testability, and separation of concerns:

1. **Ingestion Layer (`app/ingestion/`)**: Responsible for taking raw documents, parsing them (using LangChain loaders), chunking them into semantic pieces, and sending them to the vector store.
2. **Retrieval Layer (`app/retrieval/`)**: Handles embedding generation and vector database interaction (ChromaDB). This layer ensures that relevant context can be fetched semantically.
3. **Generation Layer (`app/generation/`)**: Contains the RAG logic to take user queries and retrieved context, and synthesize an accurate answer with citations.
4. **Agent/Workflow Layer (`app/graph/`)**: Implements the control flow using LangGraph. It routes user queries through retrieval, generation, and critique nodes, supporting retry loops if the critic determines the context is insufficient.
5. **Evaluation Layer (`app/evaluation/`)**: An independent harness to test the RAG pipeline with deterministic metrics (Recall@K, MRR, Relevance) to quantify improvements and track regressions over time.
6. **API Layer (`app/api/`)**: A FastAPI implementation to expose the intelligence platform to end users.

## Design Decisions & Learnings

- **Explicit State Management**: By using LangGraph, we've moved away from opaque agent loops to an explicit state machine. This makes it trivial to inject a "critic" that forces a retry when hallucination risks are high.
- **Evaluation-Driven Development (BDD)**: The platform was built test-first. Every component has BDD scenarios enforcing its behavior, which ensures we don't regress when tuning embeddings or prompts.
- **Configurability**: LLM providers and embedding models are not hardcoded. They are loaded via environment variables, allowing seamless switching from local models (e.g., Ollama) to cloud providers (e.g., OpenAI/Gemini).
- **Structured Outputs**: Using Pydantic for agent tool schemas ensures the LLM generates predictable JSON, heavily reducing parsing errors.

## Running Locally

1. Create a virtual environment and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and configure your API keys.
3. Run tests:
   ```bash
   pytest
   ```
4. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```

## Docker

To run the platform using Docker Compose:

```bash
docker-compose up --build
```

## Project Status

This project is actively maintained and serves as a blueprint for production RAG systems.
