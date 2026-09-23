import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Specify the local directory for Chroma
CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

def get_embeddings():
    # Make sure your ollama server is running and the embedding model is pulled, e.g., 'nomic-embed-text' or 'llama3'
    # Defaulting to llama3 embeddings as it's common for ollama, change if you have a specific embedding model
    return OllamaEmbeddings(model="llama3")

def ingest_documents():
    # 1. Load documents
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created '{DATA_PATH}' directory. Please add some documents there and run again if it's empty.")
        
    print(f"Loading documents from {DATA_PATH}...")
    # Use glob to load all types of files, unstructured will handle them if installed
    loader = DirectoryLoader(DATA_PATH, glob="**/*.*", show_progress=True)
    documents = loader.load()
    
    if not documents:
        print("No documents found to ingest.")
        return None

    # 2. Chunk documents
    print(f"Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")

    # 3. Create Embeddings & Store in Vector DB
    print("Creating embeddings and storing in Chroma...")
    embeddings = get_embeddings()
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Ingestion complete. Data stored in '{CHROMA_PATH}'")
    return db

def get_retriever():
    """Returns a retriever interface to the Chroma DB"""
    embeddings = get_embeddings()
    # Initialize Chroma from the persisted directory
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # K=3 is the number of documents to retrieve
    return db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    ingest_documents()
