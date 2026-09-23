import time
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from ingest import get_retriever

# ==========================================
# 1. State Definition
# ==========================================
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    metrics: dict # Store latency, accuracy, hallucination scores

# ==========================================
# 2. LLM Setup
# ==========================================
# Using native ChatOllama for the nodes
llm = ChatOllama(model="llama3", temperature=0)
# Use JSON mode for LLM that supports it, or standard text for evaluation
eval_llm = ChatOllama(model="llama3", temperature=0)

# ==========================================
# 3. Pydantic Models for Evaluation
# ==========================================
class GradeDocuments(BaseModel):
    """Boolean score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucination(BaseModel):
    """Boolean score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Boolean score for answer addressing the question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# ==========================================
# 4. Node Functions
# ==========================================
def retrieve(state: AgentState):
    """Retrieve documents"""
    print("---RETRIEVE---")
    question = state["question"]
    start_time = time.time()
    
    retriever = get_retriever()
    documents = retriever.invoke(question)
    
    latency = time.time() - start_time
    
    metrics = state.get("metrics", {})
    metrics["retrieval_latency"] = f"{latency:.2f}s"
    
    return {"documents": documents, "question": question, "metrics": metrics}

def generate(state: AgentState):
    """Generate answer"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Prompt
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:""",
        input_variables=["question", "context"],
    )
    
    # Chain
    rag_chain = prompt | llm
    
    start_time = time.time()
    # Execute
    generation = rag_chain.invoke({"context": documents, "question": question})
    latency = time.time() - start_time
    
    metrics = state.get("metrics", {})
    metrics["generation_latency"] = f"{latency:.2f}s"
    
    return {"documents": documents, "question": question, "generation": generation.content, "metrics": metrics}

def grade_documents(state: AgentState):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    parser = PydanticOutputParser(pydantic_object=GradeDocuments)
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        {format_instructions}""",
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | eval_llm | parser
    
    filtered_docs = []
    relevant_count = 0
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            relevant_count += 1
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
            
    metrics = state.get("metrics", {})
    metrics["retrieval_score"] = f"{relevant_count}/{len(documents)} relevant"
    
    return {"documents": filtered_docs, "question": question, "metrics": metrics}

def transform_query(state: AgentState):
    """Transform the query to produce a better question."""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question:""",
        input_variables=["question"],
    )
    
    chain = prompt | llm
    better_question = chain.invoke({"question": question})
    
    return {"documents": state["documents"], "question": better_question.content, "metrics": state.get("metrics", {})}


# ==========================================
# 5. Conditional Edges
# ==========================================
def decide_to_generate(state: AgentState):
    """Decide whether to generate an answer, or re-generate a query."""
    print("---DECIDE TO GENERATE---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state: AgentState):
    """Determines whether the generation is grounded in the document and answers question."""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    metrics = state.get("metrics", {})
    
    # Check Hallucination
    hallucination_parser = PydanticOutputParser(pydantic_object=GradeHallucination)
    hallucination_prompt = PromptTemplate(
        template="""You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the LLM generation: {generation}
        {format_instructions}""",
        input_variables=["documents", "generation"],
        partial_variables={"format_instructions": hallucination_parser.get_format_instructions()},
    )
    
    hallucination_chain = hallucination_prompt | eval_llm | hallucination_parser
    
    score = hallucination_chain.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        metrics["hallucination_score"] = "Pass (Grounded)"
        
        # Check Answer Accuracy
        print("---GRADE GENERATION vs QUESTION---")
        answer_parser = PydanticOutputParser(pydantic_object=GradeAnswer)
        answer_prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer addresses / resolves a question. \n 
            Here is the question: {question} \n
            Here is the answer: {generation} \n
            {format_instructions}""",
            input_variables=["question", "generation"],
            partial_variables={"format_instructions": answer_parser.get_format_instructions()},
        )
        answer_chain = answer_prompt | eval_llm | answer_parser
        answer_score = answer_chain.invoke({"question": question, "generation": generation})
        
        if answer_score.binary_score == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            metrics["accuracy_score"] = "Pass (Accurate)"
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            metrics["accuracy_score"] = "Fail (Inaccurate)"
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        metrics["hallucination_score"] = "Fail (Hallucination)"
        return "not supported"

# ==========================================
# 6. Graph Compilation
# ==========================================
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    
    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    
    # Compile
    app = workflow.compile()
    return app
