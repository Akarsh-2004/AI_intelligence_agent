from pydantic import BaseModel, Field
from typing import List

class CritiqueResult(BaseModel):
    supported: bool = Field(description="Whether the answer is supported by the retrieved context.")
    confidence: float = Field(description="Confidence score between 0 and 1.")
    missing_information: List[str] = Field(description="List of information that is missing from context to answer fully.")
    reasoning: str = Field(description="Reasoning for the critique decision.")
