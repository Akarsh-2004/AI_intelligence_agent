import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

def get_embeddings_model() -> Embeddings:
    model_name = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)
