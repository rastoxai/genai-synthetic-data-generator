import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the functions to be tested from your Streamlit app file
# Assuming your Streamlit app file is named 'app.py'
from app import parse_schema_string, generate_initial_sample_data, synthesize_data_with_sdv


# --- Tests for parse_schema_string ---

def test_parse_schema_string_comma_separated():
    """Tests parsing a comma-separated schema string."""
    schema_str = "col1, col2,col3"
    expected_columns = ["col1", "col2", "col3"]
    assert parse_schema_string(schema_str) == expected_columns

def test_parse_schema_string_space_separated():
    """Tests parsing a space-separated schema string."""
    schema_str = "colA colB  colC"
    expected_columns = ["colA", "colB", "colC"]
    assert parse_schema_string(schema_str) == expected_columns

def test_parse_schema_string_mixed_separators():
    """Tests parsing a schema string with mixed commas and spaces."""
    schema_str = "colX,colY colZ , colW"
    expected_columns = ["colX", "colY", "colZ", "colW"]
    assert parse_schema_string(schema_str) == expected_columns

def test_parse_schema_string_leading_trailing_spaces():
    """Tests parsing a schema string with leading/trailing spaces."""
    schema_str = "  col1,col2   "
    expected_columns = ["col1", "col2"]
    assert parse_schema_string(schema_str) == expected_columns

def test_parse_schema_string_empty_string():
    """Tests parsing an empty schema string."""
    schema_str = ""
    expected_columns = []
    assert parse_schema_string(schema_str) == expected_columns

def test_parse_schema_string_only_spaces_commas():
    """Tests parsing a schema string with only spaces and commas."""
    schema_str = " ,  , "
    expected_columns = []
    assert parse_schema_string(schema_str) == expected_columns

# --- Tests for generate_initial_sample_data ---
# Note: Testing LLM output deterministically is hard.
# These tests focus on the function's structure and error handling.

@patch('app.genai.GenerativeModel') # Patch the Gemini model call
def test_generate_initial_sample_data_returns_dict_of_dfs(mock_generative_model):
    """Tests that the function returns a dictionary of DataFrames on success."""
    # Mock the Gemini response to simulate successful data generation
    mock_response = MagicMock()
    # Simulate a plausible Gemini response format
    mock_response.text = """
Table1:
value1_1,value1_2
value2_1,value2_2

Table2:
valueA_1,valueA_2,valueA_3
valueB_1,valueB_2,valueB_3
"""
    mock_generative_model.return_value.generate_content.return_value = mock_response

    table_schemas = {
        "Table1": ["col1", "col2"],
        "Table2": ["colA", "colB", "colC"]
    }

    # Patch st.info to prevent Streamlit calls during testing
    with patch('app.st.info'), patch('app.st.warning'), patch('app.st.error'):
        initial_data = generate_initial_sample_data(table_schemas)

    assert isinstance(initial_data, dict)
    assert list(initial_data.keys()) == ["Table1", "Table2"] # Check table names
    assert all(isinstance(df, pd.DataFrame) for df in initial_data.values()) # Check types

    # Check column names of the returned DataFrames
    assert initial_data["Table1"].columns.tolist() == table_schemas["Table1"]
    assert initial_data["Table2"].columns.tolist() == table_schemas["Table2"]

    # Check number of rows (should be >= 1) based on mock data
    assert len(initial_data["Table1"]) >= 1
    assert len(initial_data["Table2"]) >= 1


@patch('app.genai.GenerativeModel')
def test_generate_initial_sample_data_handles_empty_response(mock_generative_model):
    """Tests that the function handles an empty response from Gemini."""
    mock_response = MagicMock()
    mock_response.text = "" # Simulate empty response
    mock_generative_model.return_value.generate_content.return_value = mock_response

    table_schemas = {
        "Table1": ["col1", "col2"],
        "Table2": ["colA", "colB", "colC"]
    }

    with patch('app.st.info'), patch('app.st.warning'), patch('app.st.error'):
        initial_data = generate_initial_sample_data(table_schemas)

    assert isinstance(initial_data, dict)
    # Should return empty dataframes for the requested tables
    assert list(initial_data.keys()) == ["Table1", "Table2"]
    assert all(isinstance(df, pd.DataFrame) for df in initial_data.values())
    assert all(df.empty for df in initial_data.values()) # All dataframes should be empty
    assert initial_data["Table1"].columns.tolist() == table_schemas["Table1"]
    assert initial_data["Table2"].columns.tolist() == table_schemas["Table2"]


@patch('app.genai.GenerativeModel')
def test_generate_initial_sample_data_handles_api_error(mock_generative_model):
    """Tests that the function handles an exception during the API call."""
    mock_generative_model.return_value.generate_content.side_effect = Exception("API Error")

    table_schemas = {
        "Table1": ["col1", "col2"]
    }

    with patch('app.st.info'), patch('app.st.warning'), patch('app.st.error'):
         initial_data = generate_initial_sample_data(table_schemas)

    assert isinstance(initial_data, dict)
    # Should return empty dataframes on API error
    assert list(initial_data.keys()) == ["Table1"]
    assert isinstance(initial_data["Table1"], pd.DataFrame)
    assert initial_data["Table1"].empty
    assert initial_data["Table1"].columns.tolist() == table_schemas["Table1"]


# --- Tests for synthesize_data_with_sdv ---
# Note: Testing SDV synthesis deterministically is also hard.
# These tests focus on the function's structure and error handling.

@patch('app.HMASynthesizer') # Patch the HMASynthesizer
@patch('app.MultiTableMetadata.detect_from_dataframes') # Patch metadata detection
def test_synthesize_data_with_sdv_returns_dict_of_dfs(mock_detect_metadata, mock_synthesizer):
    """Tests that the function returns a dictionary of DataFrames on success."""
    # Create mock initial data
    initial_data = {
        "Orders": pd.DataFrame({"ORDER_ID": [1, 2], "USER_ID": [10, 11]}),
        "Users": pd.DataFrame({"USER_ID": [10, 11], "Name": ["A", "B"]})
    }

    # Mock the SDV synthesizer output
    mock_synthetic_data = {
        "Orders": pd.DataFrame({"ORDER_ID": [101, 102], "USER_ID": [10, 11]}),
        "Users": pd.DataFrame({"USER_ID": [10, 11], "Name": ["C", "D"]})
    }
    mock_synthesizer.return_value.sample.return_value = mock_synthetic_data

    # Patch st.info to prevent Streamlit calls during testing
    with patch('app.st.info'), patch('app.st.warning'), patch('app.st.error'), patch('app.st.write'), patch('app.st.json'):
        synthetic_data = synthesize_data_with_sdv(initial_data, scale=2)

    assert isinstance(synthetic_data, dict)
    assert list(synthetic_data.keys()) == ["Orders", "Users"] # Check table names
    assert all(isinstance(df, pd.DataFrame) for df in synthetic_data.values()) # Check types

    # Check column names of the returned DataFrames
    assert synthetic_data["Orders"].columns.tolist() == initial_data["Orders"].columns.tolist()
    assert synthetic_data["Users"].columns.tolist() == initial_data["Users"].columns.tolist()

    # Check number of rows (should match the mocked output size)
    assert len(synthetic_data["Orders"]) == len(mock_synthetic_data["Orders"])
    assert len(synthetic_data["Users"]) == len(mock_synthetic_data["Users"])


def test_synthesize_data_with_sdv_handles_empty_initial_data():
    """Tests that the function handles empty initial data."""
    initial_data = {} # Empty dictionary

    with patch('app.st.info'), patch('app.st.warning'), patch('app.st.error'), patch('app.st.write'), patch('app.st.json'):
        synthetic_data = synthesize_data_with_sdv(initial_data, scale=10)

    assert isinstance(synthetic_data, dict)
    assert not synthetic_data # Should return an empty dictionary

def test_synthesize_data_with_sdv_handles_all_empty_dfs_in_initial_data():
    """Tests that the function handles initial data with only empty DataFrames."""
    initial_data = {
        "Table1": pd.DataFrame(columns=["col1", "col2"]),
        "Table2": pd.DataFrame(columns=["colA", "colB", "colC"])
    }

    with patch('app.st.info'), patch('app.st.warning'), patch('app.st.error'), patch('app.st.write'), patch('app.st.json'):
        synthetic_data = synthesize_data_with_sdv(initial_data, scale=10)

    assert isinstance(synthetic_data, dict)
    assert not synthetic_data # Should return an empty dictionary

@patch('app.MultiTableMetadata.detect_from_dataframes')
def test_synthesize_data_with_sdv_handles_sdv_error(mock_detect_metadata):
    """Tests that the function handles an exception during SDV processing."""
    # Simulate an error during metadata detection or synthesis
    mock_detect_metadata.side_effect = Exception("SDV Error")

    initial_data = {
        "Orders": pd.DataFrame({"ORDER_ID": [1, 2], "USER_ID": [10, 11]})
    }

    with patch('app.st.info'), patch('app.st.warning'), patch('app.st.error'), patch('app.st.write'), patch('app.st.json'):
        synthetic_data = synthesize_data_with_sdv(initial_data, scale=10)

    assert isinstance(synthetic_data, dict)
    assert not synthetic_data # Should return an empty dictionary on error
