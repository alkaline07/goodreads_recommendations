"""
Test cases for model_validation.py
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from src.model_validation import BigQueryModelValidator


class TestBigQueryModelValidator:
    """Test cases for BigQueryModelValidator class"""
    
    @patch('src.model_validation.bigquery.Client')
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        validator = BigQueryModelValidator()
        
        assert validator.project_id == "test-project"
        assert validator.dataset_id == "books"
        assert "goodreads_train_set" in validator.train_table
    
    @patch('src.model_validation.bigquery.Client')
    def test_evaluate_split(self, mock_client_class):
        """Test evaluating a split"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'mean_absolute_error': [0.5],
            'rmse': [0.6]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        validator = BigQueryModelValidator()
        validator.client = mock_client
        
        result = validator.evaluate_split(
            "test-model",
            "val",
            "test-table"
        )
        
        assert not result.empty
        assert 'mean_absolute_error' in result.columns
    
    @patch('src.model_validation.bigquery.Client')
    @patch('src.model_validation.mlflow.start_run')
    def test_validate_model_pass(self, mock_mlflow_run, mock_client_class):
        """Test model validation that passes"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'rmse': [2.5]  # Below threshold of 3.0
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_mlflow_context = Mock()
        mock_mlflow_run.return_value.__enter__.return_value = mock_mlflow_context
        mock_mlflow_run.return_value.__exit__.return_value = None
        
        validator = BigQueryModelValidator()
        validator.client = mock_client
        
        result = validator.validate_model(
            "test-model",
            "test-label"
        )
        
        assert result is True
    
    @patch('src.model_validation.bigquery.Client')
    @patch('src.model_validation.mlflow.start_run')
    def test_validate_model_fail(self, mock_mlflow_run, mock_client_class):
        """Test model validation that fails"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'rmse': [3.5]  # Above threshold of 3.0
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_mlflow_context = Mock()
        mock_mlflow_run.return_value.__enter__.return_value = mock_mlflow_context
        mock_mlflow_run.return_value.__exit__.return_value = None
        
        validator = BigQueryModelValidator()
        validator.client = mock_client
        
        result = validator.validate_model(
            "test-model",
            "test-label"
        )
        
        assert result is False
    
    @patch('src.model_validation.bigquery.Client')
    def test_save_json_report(self, mock_client_class):
        """Test saving JSON report"""
        import os
        import tempfile
        
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        validator = BigQueryModelValidator()
        validator.client = mock_client
        validator.project_id = "test-project"
        validator.dataset_id = "books"
        
        results = {
            "train": pd.DataFrame({'mae': [0.5]}),
            "val": pd.DataFrame({'mae': [0.6]}),
            "test": pd.DataFrame({'mae': [0.7]})
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.model_validation.os.path.abspath', return_value=tmpdir):
                validator.save_json_report("test-label", "test-model", results)
                # Should not raise exception
                assert True

