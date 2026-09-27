import os
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader

class DocumentLoader:
    @staticmethod
    def load(file_path: str) -> List[Any]:
        ext = Path(file_path).suffix.lower()
        if ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()
