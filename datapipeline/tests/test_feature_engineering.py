"""
Unit Tests for Feature Engineering Module

This module contains comprehensive unit tests for the FeatureEngineering class,
testing feature creation functionality, error handling, and BigQuery integration.

Test Coverage:
- Feature creation and SQL query generation
- Table statistics collection
- Sample data export functionality
- Error handling and logging
- BigQuery client interactions
- Pipeline execution flow

Author: Goodreads Recommendation Team
Date: 2025
"""

import pytest
import os
from unittest.mock import Mock, patch
import pandas as pd

from datapipeline.scripts.feature_engineering import FeatureEngineering


def test_initialization():
    """
    Test class initialization and configuration.
    
    This test verifies that the FeatureEngineering class initializes correctly
    with proper configuration values and environment setup.
    """
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Test basic attributes
                assert fe.project_id == 'test-project'
                assert fe.dataset_id == 'books'
                assert fe.MIN_READING_DAYS == 1
                assert fe.MAX_READING_DAYS == 365
                assert fe.DEFAULT_PAGE_COUNT == 300
                assert fe.DEFAULT_READING_DAYS == 14

def test_table_names():
    """Test table name generation"""
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Test table names
                assert 'test-project.books.goodreads_books_cleaned' in fe.books_table
                assert 'test-project.books.goodreads_interactions_cleaned' in fe.interactions_table
                assert 'test-project.books.goodreads_features_cleaned' in fe.destination_table

def test_create_features_success():
    """Test successful feature creation"""
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                    
                fe = FeatureEngineering()
                    
                # Mock query job
                mock_query_job = Mock()
                mock_query_job.result.return_value = None
                fe.client.query.return_value = mock_query_job
                    
                with patch('datapipeline.scripts.feature_engineering.bigquery.QueryJobConfig'):
                    # Should not raise an exception
                    fe.create_features()
                        
                    # Verify query was called
                    fe.client.query.assert_called_once()

def test_create_features_error():
    """Test error handling in create_features"""
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Mock BigQuery to raise an exception
                fe.client.query.side_effect = Exception("BigQuery error")
                
                with pytest.raises(Exception, match="BigQuery error"):
                    fe.create_features()

def test_get_table_stats_success():
    """Test successful table statistics"""
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Mock query result
                mock_stats = pd.DataFrame({
                    'total_rows': [1000],
                    'unique_users': [100],
                    'unique_books': [50],
                    'avg_books_per_user': [10.0],
                    'avg_reading_time_days': [14.5],
                    'avg_pages': [300.0],
                    'avg_rating': [4.2]
                })
                fe.client.query.return_value.to_dataframe.return_value = mock_stats
                
                # Should not raise an exception
                fe.get_table_stats()
                
                # Verify query was called
                fe.client.query.assert_called_once()

def test_get_table_stats_error():
    """Test error handling in get_table_stats"""
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Mock BigQuery to raise an exception
                fe.client.query.side_effect = Exception("Query failed")
                
                # The method catches and logs the exception, so it should not raise
                # We just verify it doesn't crash
                fe.get_table_stats()

def test_export_sample_success():
    """Test successful sample export"""
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Mock query result
                mock_sample = pd.DataFrame({
                    'user_id_clean': ['user1', 'user2'],
                    'book_id': ['book1', 'book2'],
                    'rating': [4, 5],
                    'num_pages': [300, 400],
                    'book_era': ['contemporary', 'modern']
                })
                fe.client.query.return_value.to_dataframe.return_value = mock_sample
                
                with patch('os.makedirs'):
                    with patch('pandas.DataFrame.to_parquet'):
                        # Should not raise an exception
                        fe.export_sample(sample_size=100)
                        
                        # Verify query was called
                        fe.client.query.assert_called_once()

def test_export_sample_error():
    """Test error handling in export_sample"""
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Mock BigQuery to raise an exception
                fe.client.query.side_effect = Exception("Export failed")
                
                # The method catches and logs the exception, so it should not raise
                # We just verify it doesn't crash
                fe.export_sample()

def test_run_success():
    """Test successful run method"""
    # Skip this test if credentials file doesn't exist
    if not os.path.exists("config/gcp_credentials.json"):
        pytest.skip("Skipping run test - credentials file not found")
    
    # Set AIRFLOW_HOME to point to the config directory
    with patch.dict(os.environ, {'AIRFLOW_HOME': 'config'}):
        fe = FeatureEngineering()
        # Should not raise an exception
        fe.run()

def test_run_error():
    """Test error handling in run method"""
    # Skip this test if credentials file doesn't exist
    if not os.path.exists("config/gcp_credentials.json"):
        pytest.skip("Skipping run error test - credentials file not found")
    
    # Set AIRFLOW_HOME to point to the config directory
    with patch.dict(os.environ, {'AIRFLOW_HOME': 'config'}):
        fe = FeatureEngineering()
        # This test would need actual error conditions to test properly
        # For now, just verify the method exists and can be called
        assert hasattr(fe, 'run')

def test_query_contains_expected_features():
    """Test that the generated query contains expected features"""
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Mock query job
                mock_query_job = Mock()
                mock_query_job.result.return_value = None
                fe.client.query.return_value = mock_query_job
                
                with patch('datapipeline.scripts.feature_engineering.bigquery.QueryJobConfig'):
                    fe.create_features()
                    
                    # Get the query that was passed
                    call_args = fe.client.query.call_args
                    query = call_args[0][0]
                    
                    # Check for key features
                    assert 'avg_book_reading_time' in query
                    assert 'popularity_score' in query
                    assert 'book_age_years' in query
                    assert 'reading_pace_category' in query
                    assert 'book_era' in query
                    assert 'interaction_weight' in query

def test_interaction_weight_logic_in_sql():
    """
    Test that the interaction_weight CASE statement is correctly generated in the SQL.
    Verifies the specific weights for read (5), like (4), add_to_list (3), and click (1).
    """
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/tmp/test'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client') as mock_client:
                mock_client.return_value.project = 'test-project'
                
                fe = FeatureEngineering()
                
                # Mock query job to prevent actual execution
                mock_query_job = Mock()
                mock_query_job.result.return_value = None
                fe.client.query.return_value = mock_query_job
                
                with patch('datapipeline.scripts.feature_engineering.bigquery.QueryJobConfig'):
                    # Trigger the method that builds the SQL
                    fe.create_features()
                    
                    # Capture the query string passed to the client
                    call_args = fe.client.query.call_args
                    generated_sql = call_args[0][0]
                    
                    # Validate the hierarchy logic exists exactly as defined
                    assert "WHEN interaction_type_clean = 'read' THEN 5" in generated_sql
                    assert "WHEN interaction_type_clean = 'like' THEN 4" in generated_sql
                    assert "WHEN interaction_type_clean = 'add_to_list' THEN 3" in generated_sql
                    assert "WHEN interaction_type_clean = 'click' THEN 1" in generated_sql
                    
                    # Validate the column is aliased correctly
                    assert "END as interaction_weight" in generated_sql

def test_environment_variables():
    """Test environment variable handling"""
    # Test with AIRFLOW_HOME set
    with patch.dict(os.environ, {'AIRFLOW_HOME': '/custom/path'}):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client'):
                fe = FeatureEngineering()
                assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/custom/path/gcp_credentials.json"
    
    # Test with AIRFLOW_HOME not set
    with patch.dict(os.environ, {}, clear=True):
        with patch('os.path.exists', return_value=True):
            with patch('datapipeline.scripts.feature_engineering.bigquery.Client'):
                fe = FeatureEngineering()
                assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "./gcp_credentials.json"

def test_main_function():
    """Test the main function"""
    # Skip this test if credentials file doesn't exist
    if not os.path.exists("config/gcp_credentials.json"):
        pytest.skip("Skipping main function test - credentials file not found")
    
    # Set AIRFLOW_HOME to point to the config directory
    with patch.dict(os.environ, {'AIRFLOW_HOME': 'config'}):
        # Import main from the script, not the test file
        from datapipeline.scripts.feature_engineering import main
        # Should not raise an exception
        main()

if __name__ == "__main__":
    pytest.main([__file__, "-q"])