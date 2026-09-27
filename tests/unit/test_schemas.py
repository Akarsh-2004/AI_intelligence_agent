from pydantic import ValidationError
import pytest
from app.agents.schemas import CritiqueResult

def test_critique_schema_validation():
    # Valid data
    valid_data = {
        "supported": True,
        "confidence": 0.9,
        "missing_information": [],
        "reasoning": "The answer directly uses the context provided."
    }
    
    critique = CritiqueResult(**valid_data)
    assert critique.supported is True
    assert critique.confidence == 0.9
    
    # Invalid data
    invalid_data = {
        "supported": "maybe",
        "confidence": "high",
        "missing_information": "none",
        "reasoning": "Looks good"
    }
    
    with pytest.raises(ValidationError):
        CritiqueResult(**invalid_data)
