import os
from typing import List, Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

class RAGGenerator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Use the following context to answer the user's question.
If the context does not contain enough information to answer, state that sufficient evidence was not found. Do not invent supporting facts.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

    def generate(self, query: str, documents: List[Any]) -> Dict[str, Any]:
        if not documents:
            return {
                "answer": "Sufficient evidence was not found to answer the question.",
                "sources": []
            }

        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Prepare the chain
        chain = self.prompt | self.llm
        
        # Generate the answer
        response = chain.invoke({"context": context, "question": query})
        
        # Extract metadata for sources
        sources = [doc.metadata for doc in documents]
        
        return {
            "answer": response.content if hasattr(response, "content") else str(response),
            "sources": sources
        }
