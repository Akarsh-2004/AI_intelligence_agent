import os
from app.ingestion.pipeline import IngestionPipeline

def run_ingestion():
    print("Initializing Ingestion Pipeline...")
    pipeline = IngestionPipeline()
    
    data_dir = "./data/rag/scraped_content"
    print(f"Scanning directory: {data_dir}")
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            print(f"Ingesting {filepath}...")
            try:
                result = pipeline.ingest_document(filepath)
                print(f"Success! Indexed {result['chunks_indexed']} chunks.")
            except Exception as e:
                print(f"Failed to ingest {filepath}. Error: {e}")

if __name__ == "__main__":
    run_ingestion()
