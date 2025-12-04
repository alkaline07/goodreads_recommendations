"""
Unit Tests for Data Cleaning Module

This module contains comprehensive unit tests for the DataCleaning class,
testing data cleaning functionality, error handling, and BigQuery integration.

Test Coverage:
- Data cleaning table operations
- SQL query generation and execution
- Error handling and logging
- BigQuery client interactions
- Pipeline execution flow

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from datapipeline.scripts.data_cleaning import DataCleaning

# ---------------------------------------------------------------------
# GLOBAL FIXTURES
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """
    Ensure AIRFLOW_HOME is defined during tests.
    
    This fixture automatically sets the AIRFLOW_HOME environment variable
    for all tests to ensure proper configuration during testing.
    """
    monkeypatch.setenv("AIRFLOW_HOME", "config/")


@pytest.fixture
def mock_bq_client():
    """
    Create a mock BigQuery client for testing.
    
    Returns:
        MagicMock: Mocked BigQuery client with standard methods
    """
    mock_client = MagicMock()
    mock_query_job = MagicMock()
    mock_query_job.to_dataframe.return_value = None
    mock_query_job.result.return_value = None
    mock_client.query.return_value = mock_query_job
    mock_client.project = "test_project"
    return mock_client


@pytest.fixture
def data_cleaning_instance(mock_bq_client):
    """
    Create DataCleaning instance with mocked BigQuery client.
    
    Args:
        mock_bq_client: Mocked BigQuery client fixture
        
    Returns:
        DataCleaning: Instance with mocked dependencies
    """
    with patch("datapipeline.scripts.data_cleaning.bigquery.Client", return_value=mock_bq_client):
        dc = DataCleaning()
        dc.logger = MagicMock()
        return dc


# ---------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------

def test_clean_books_table_basic(data_cleaning_instance, mock_bq_client):
    """
    Test basic clean_books_table functionality and query generation.
    """
    # Mock column information for the test
    mock_df = MagicMock()
    mock_df.iterrows.return_value = iter([
        (0, {'column_name': 'num_pages', 'data_type': 'INT64'}),
        (1, {'column_name': 'title', 'data_type': 'STRING'})
    ])
    mock_df.__len__.return_value = 2
    # The first query call is to INFORMATION_SCHEMA, make it return the mock_df
    mock_bq_client.query.return_value.to_dataframe.return_value = mock_df

    data_cleaning_instance.clean_books_table(
        books_table_name="goodreads_books",
        books_destination="test_project.books.cleaned_books"
    )

    # The second query call (index 1) is the actual cleaning query
    query_call = mock_bq_client.query.call_args_list[1][0][0]
    assert "goodreads_books" in query_call
    assert "WHERE (num_pages >= 10 AND num_pages <= 2000)" in query_call
    assert "publication_year >= 1900" in query_call


def test_clean_table_error(data_cleaning_instance, mock_bq_client):
    """
    Test error handling in the underlying generic clean method.
    """
    # Simulate a query failure on the *first* query (INFORMATION_SCHEMA)
    mock_bq_client.query.side_effect = Exception("Query failed")

    # We expect the method to catch the exception, log it, and re-raise it
    with pytest.raises(Exception, match="Query failed"):
        data_cleaning_instance.clean_books_table(
            books_table_name="goodreads_books",
            books_destination="test_project.books.cleaned_books"
        )

    # Verify that the error was logged
    data_cleaning_instance.logger.error.assert_called_with(
        "Error cleaning table books.goodreads_books: Query failed", exc_info=True
    )


def test_clean_books_table_creates_expected_sql(data_cleaning_instance, mock_bq_client):
    """Validate generated SQL for books table contains cleaning and filtering."""
    mock_df = MagicMock()
    mock_df.iterrows.return_value = iter([
        (0, {'column_name': 'num_pages', 'data_type': 'INT64'}),
        (1, {'column_name': 'publication_year', 'data_type': 'INT64'}),
        (2, {'column_name': 'title', 'data_type': 'STRING'}),
        (3, {'column_name': 'tags', 'data_type': 'ARRAY<STRING>'}),
        (4, {'column_name': 'is_available', 'data_type': 'BOOL'})
    ])
    mock_df.__len__.return_value = 5
    mock_bq_client.query.return_value.to_dataframe.return_value = mock_df

    with patch("datapipeline.scripts.data_cleaning.bigquery.QueryJobConfig"):
        data_cleaning_instance.clean_books_table(
            books_table_name="goodreads_books",
            books_destination="test_project.books.cleaned_books"
        )

        # The clean query is the SECOND call (index 1)
        query_call = mock_bq_client.query.call_args_list[1][0][0]

        # Core checks
        assert "WHERE (num_pages >= 10" in query_call
        assert "publication_year <= 2025" in query_call
        assert "SELECT DISTINCT" in query_call
        assert "WITH" not in query_call 

def test_clean_interactions_table_excludes_views(data_cleaning_instance, mock_bq_client):
    """
    Test that clean_interactions_table applies the filter to exclude 'view' interaction types.
    """
    # 1. Mock the schema response so the column loop functions correctly
    mock_df = MagicMock()
    mock_df.iterrows.return_value = iter([
        (0, {'column_name': 'interaction_type', 'data_type': 'STRING'}),
        (1, {'column_name': 'book_id', 'data_type': 'INT64'}),
        (2, {'column_name': 'user_id', 'data_type': 'INT64'})
    ])
    mock_df.__len__.return_value = 3
    
    # The first query call is to INFORMATION_SCHEMA
    mock_bq_client.query.return_value.to_dataframe.return_value = mock_df

    # 2. Run the method
    with patch("datapipeline.scripts.data_cleaning.bigquery.QueryJobConfig"):
        data_cleaning_instance.clean_interactions_table(
            interactions_table_name="goodreads_interactions",
            interactions_destination="test_project.books.cleaned_interactions",
            books_destination="test_project.books.cleaned_books"
        )

    # 3. Retrieve the generated SQL
    # Call 0: Information Schema
    # Call 1: The cleaning query (This is the one we want to check)
    # Call 2: The JOIN/Filter query
    cleaning_query_call = mock_bq_client.query.call_args_list[1][0][0]

    # 4. Assert the filter clause exists
    assert "WHERE interaction_type != 'view'" in cleaning_query_call
    assert "SELECT DISTINCT" in cleaning_query_call
    assert "test_project.books.goodreads_interactions" in cleaning_query_call

def test_clean_interactions_table_filters_against_books(data_cleaning_instance, mock_bq_client):
    """
    Test that clean_interactions_table first cleans and then filters against the books table.
    """
    mock_df = MagicMock()
    mock_df.iterrows.return_value = iter([
        (0, {'column_name': 'book_id', 'data_type': 'INT64'}),
        (1, {'column_name': 'user_id', 'data_type': 'STRING'})
    ])
    mock_df.__len__.return_value = 2
    mock_bq_client.query.return_value.to_dataframe.return_value = mock_df

    with patch("datapipeline.scripts.data_cleaning.bigquery.QueryJobConfig"):
        data_cleaning_instance.clean_interactions_table(
            interactions_table_name="goodreads_interactions",
            interactions_destination="test_project.books.cleaned_interactions",
            books_destination="test_project.books.cleaned_books"
        )

    # Verify it was called three times
    # 1. INFORMATION_SCHEMA
    # 2. Generic Clean Query
    # 3. Filtering JOIN Query
    assert mock_bq_client.query.call_count == 3
    
    # Check the second query (generic clean)
    query_call_1 = mock_bq_client.query.call_args_list[1][0][0]
    assert "goodreads_interactions" in query_call_1

    # Check the third query (filtering join)
    query_call_2 = mock_bq_client.query.call_args_list[2][0][0]
    assert "test_project.books.cleaned_interactions" in query_call_2
    assert "test_project.books.cleaned_books" in query_call_2
    assert "INNER JOIN" in query_call_2
    assert "t1.book_id = t2.book_id" in query_call_2


def test_run_pipeline(data_cleaning_instance, mock_bq_client):
    """Ensure run executes all cleaning steps correctly."""
    # Patch the methods called by run()
    with patch.object(data_cleaning_instance, "clean_books_table") as mock_clean_books, \
         patch.object(data_cleaning_instance, "clean_interactions_table") as mock_clean_interactions:
        
        mock_sample_df = MagicMock()
        # Mock the return value for the sample queries
        mock_bq_client.query.return_value.to_dataframe.return_value = mock_sample_df

        data_cleaning_instance.run()
        
        # Verify the main cleaning steps were called
        mock_clean_books.assert_called_once()
        mock_clean_interactions.assert_called_once()
        
        # Check that the sample queries were called
        # These queries are *not* patched, so they hit the mock_bq_client
        sample_query_calls = [
            call for call in mock_bq_client.query.call_args_list 
            if "LIMIT 5" in call[0][0]
        ]
        assert len(sample_query_calls) == 2


def test_main_executes(monkeypatch):
    """Test that main() runs without crashing."""
    from datapipeline.scripts import data_cleaning

    fake_client = MagicMock()
    fake_client.project = "test_project"
    monkeypatch.setattr(data_cleaning.bigquery, "Client", lambda: fake_client)

    mock_run = MagicMock()
    monkeypatch.setattr(data_cleaning.DataCleaning, "run", mock_run)
    data_cleaning.main()
    mock_run.assert_called_once()


def test_run_handles_exceptions(data_cleaning_instance):
    """Ensure run() propagates exceptions from cleaning methods."""
    # Patch clean_books_table to raise an exception
    with patch.object(data_cleaning_instance, "clean_books_table", side_effect=Exception("Query failed")):
        
        # We don't need to patch the other methods
        
        # Run the method and assert it raises the exception
        with pytest.raises(Exception, match="Query failed"):
            data_cleaning_instance.run()
        
        # Verify the logger was *not* called by the run() method's
        # (now non-existent) try/except block
        # The logger *inside* clean_books_table would be called, but since
        # we patched the *whole method*, no logger calls are made by default.
        data_cleaning_instance.logger.error.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])