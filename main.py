import sys
from ingest import ingest_documents
from graph import build_agent_graph

def run_agent(query: str):
    print(f"\n======================================")
    print(f"QUERY: {query}")
    print(f"======================================\n")
    
    app = build_agent_graph()
    
    inputs = {"question": query, "metrics": {}}
    
    # Run the graph
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished node: {key}")
            
    final_state = value
    
    print("\n======================================")
    print("FINAL ANSWER:")
    print("======================================")
    print(final_state.get("generation", "No answer generated."))
    
    print("\n======================================")
    print("PERFORMANCE METRICS:")
    print("======================================")
    metrics = final_state.get("metrics", {})
    for k, v in metrics.items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Engineering Intelligence Platform (RAG + LangGraph Agent)")
    parser.add_argument("--ingest", action="store_true", help="Run the document ingestion pipeline first")
    parser.add_argument("--query", type=str, help="The question to ask the agent", default="What are the key concepts of LangGraph?")
    
    args = parser.parse_args()
    
    if args.ingest:
        print("Starting ingestion pipeline...")
        ingest_documents()
        
    run_agent(args.query)
