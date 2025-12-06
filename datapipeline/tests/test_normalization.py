"""
Unit Tests for Data Normalization Module

This module contains comprehensive unit tests for the GoodreadsNormalization class,
testing normalization functionality, error handling, and BigQuery integration.

Test Coverage:
- Log transformation operations
- User-centered rating normalization
- Error handling and logging
- BigQuery client interactions
- Pipeline execution flow

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from datapipeline.scripts.normalization import GoodreadsNormalization


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
    mock_query_job.result.return_value = None
    mock_client.query.return_value = mock_query_job
    mock_client.project = "test_project"
    return mock_client


@pytest.fixture
def normalization_instance(mock_bq_client):
    """
    Create GoodreadsNormalization instance with mocked BigQuery client.
    
    Args:
        mock_bq_client: Mocked BigQuery client fixture
        
    Returns:
        GoodreadsNormalization: Instance with mocked dependencies
    """
    with patch("datapipeline.scripts.normalization.bigquery.Client", return_value=mock_bq_client):
        gn = GoodreadsNormalization()
        gn.logger = MagicMock()
        return gn


def test_initialization(normalization_instance):
    """Ensure initialization sets credentials and attributes correctly."""
    assert normalization_instance.project_id == "test_project"
    assert normalization_instance.dataset_id == "books"
    assert "goodreads_features_cleaned_staging" in normalization_instance.table
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"].endswith("gcp_credentials.json")


def test_log_transform_features_success(normalization_instance, mock_bq_client):
    """Ensure log_transform_features executes query successfully."""
    normalization_instance.log_transform_features()

    query_call = mock_bq_client.query.call_args[0][0]
    assert "LN(popularity_score + 1)" in query_call
    assert "description_length" in query_call
    normalization_instance.logger.info.assert_any_call("Applying log transformations to skewed features...")
    mock_bq_client.query.assert_called_once()


def test_log_transform_features_error(normalization_instance, mock_bq_client):
    """Ensure exception in log_transform_features is logged and raised."""
    mock_bq_client.query.side_effect = Exception("Log transform failed")

    with pytest.raises(Exception, match="Log transform failed"):
        normalization_instance.log_transform_features()

    normalization_instance.logger.error.assert_called()


def test_normalize_user_ratings_success(normalization_instance, mock_bq_client):
    """Ensure normalize_user_ratings executes both ALTER and UPDATE queries."""
    normalization_instance.normalize_user_ratings()
    assert mock_bq_client.query.call_count == 2
    normalization_instance.logger.info.assert_any_call("Applying user-centered rating normalization...")


def test_normalize_user_ratings_error(normalization_instance, mock_bq_client):
    """Ensure exception in normalize_user_ratings is logged and raised."""
    mock_bq_client.query.side_effect = Exception("Normalization failed")

    with pytest.raises(Exception, match="Normalization failed"):
        normalization_instance.normalize_user_ratings()

    normalization_instance.logger.error.assert_called()


def test_run_pipeline_success(normalization_instance):
    """Ensure run executes both steps successfully."""
    with patch.object(normalization_instance, "log_transform_features") as mock_log, \
         patch.object(normalization_instance, "normalize_user_ratings") as mock_norm:
        mock_log.return_value = None
        mock_norm.return_value = None

        normalization_instance.run()

        mock_log.assert_called_once()
        mock_norm.assert_called_once()
        normalization_instance.logger.info.assert_any_call("Good Reads Normalization Pipeline")



def test_run_pipeline_error(normalization_instance):
    """Ensure run raises exception if a step fails."""
    with patch.object(normalization_instance, "log_transform_features", side_effect=Exception("Log error")):
        with pytest.raises(Exception, match="Log error"):
            normalization_instance.run()

def test_environment_variables(mock_bq_client):
    """Test environment variable handling."""
    from datapipeline.scripts.normalization import GoodreadsNormalization


    with patch.dict(os.environ, {'AIRFLOW_HOME': '/custom/path'}):
        with patch("datapipeline.scripts.normalization.bigquery.Client", return_value=mock_bq_client):
            gn = GoodreadsNormalization()
            assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/custom/path/gcp_credentials.json"


    with patch.dict(os.environ, {}, clear=True):
        with patch("datapipeline.scripts.normalization.bigquery.Client", return_value=mock_bq_client):
            gn = GoodreadsNormalization()
            assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "./gcp_credentials.json"


def test_main_executes(monkeypatch):
    """Ensure main() runs the pipeline."""
    from datapipeline.scripts import normalization

    # Stub BigQuery client to avoid real credential lookup
    fake_client = MagicMock()
    fake_client.project = "test_project"
    monkeypatch.setattr(normalization.bigquery, "Client", lambda: fake_client)

    mock_run = MagicMock()
    monkeypatch.setattr(normalization.GoodreadsNormalization, "run", mock_run)
    normalization.main()
    mock_run.assert_called_once()


def test_normalize_user_ratings_query_order(normalization_instance, mock_bq_client):
    """Ensure ALTER runs before UPDATE in normalize_user_ratings."""
    normalization_instance.normalize_user_ratings()

    # Collect SQL strings in the order they were called
    called_sql = [call.args[0] for call in mock_bq_client.query.call_args_list]
    assert len(called_sql) == 2
    assert "ALTER COLUMN rating SET DATA TYPE FLOAT64" in called_sql[0]
    assert "SET rating = rating - avg_rating_given" in called_sql[1]


def test_log_transform_features_contains_edge_case_clauses(normalization_instance, mock_bq_client):
    """Ensure CASE clauses for non-positive values are included in the SQL."""
    normalization_instance.log_transform_features()

    sql = mock_bq_client.query.call_args[0][0]
    # Check key CASE clauses to handle zeros/negatives
    assert "num_books_read = CASE" in sql
    assert "WHEN num_books_read > 0 THEN CAST(LN(num_books_read + 1) AS INT64)" in sql
    assert "user_days_to_read = CASE" in sql
    assert "WHEN user_days_to_read > 0 THEN CAST(LN(user_days_to_read) AS INT64)" in sql

if __name__ == "__main__":
    pytest.main([__file__, "-q"])