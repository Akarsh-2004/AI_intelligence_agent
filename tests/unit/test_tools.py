from app.tools.tools import calculate

def test_calculate_tool():
    result = calculate.invoke({"expression": "5 * 3"})
    assert result == "15"
    
    # Test safe eval fallback / error handling
    result_error = calculate.invoke({"expression": "import os; os.system('echo bad')"})
    assert "Error" in result_error
