"""
Test cases for bq_model_training.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from src.bq_model_training import BigQueryMLModelTraining, safe_mlflow_log


class TestSafeMLflowLog:
    """Test safe_mlflow_log function"""
    
    def test_safe_mlflow_log_success(self):
        """Test successful MLflow logging"""
        mock_func = Mock(return_value="success")
        result = safe_mlflow_log(mock_func, "arg1", kwarg1="value1")
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
    
    def test_safe_mlflow_log_failure(self):
        """Test MLflow logging with exception"""
        mock_func = Mock(side_effect=Exception("MLflow error"))
        result = safe_mlflow_log(mock_func, "arg1")
        assert result is None


class TestBigQueryMLModelTraining:
    """Test cases for BigQueryMLModelTraining class"""
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch.dict(os.environ, {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        
        assert trainer.project_id == "test-project"
        assert trainer.dataset_id == "books"
        assert trainer.train_table == "test-project.books.goodreads_train_set"
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_get_feature_columns(self, mock_client_class):
        """Test getting feature columns"""
        mock_client = Mock()
        mock_table = Mock()
        # Create mock schema fields with .name attribute
        field1 = Mock()
        field1.name = 'user_id_clean'
        field2 = Mock()
        field2.name = 'book_id'
        field3 = Mock()
        field3.name = 'rating'
        field4 = Mock()
        field4.name = 'num_books_read'
        field5 = Mock()
        field5.name = 'avg_rating_given'
        mock_table.schema = [field1, field2, field3, field4, field5]
        mock_client.get_table.return_value = mock_table
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        
        features = trainer.get_feature_columns()
        
        assert 'num_books_read' in features
        assert 'avg_rating_given' in features
        assert 'user_id_clean' not in features
        assert 'rating' not in features
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_get_feature_columns_error(self, mock_client_class):
        """Test get_feature_columns with error handling"""
        mock_client = Mock()
        mock_client.get_table.side_effect = Exception("Table not found")
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        
        features = trainer.get_feature_columns()
        
        # Should return default feature list
        assert isinstance(features, list)
        assert len(features) > 0
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.time.sleep')
    def test_check_and_cleanup_existing_models(self, mock_sleep, mock_client_class):
        """Test checking and cleanup of existing models"""
        mock_client = Mock()
        mock_client.query.return_value.to_dataframe.return_value = Mock(empty=False)
        mock_client.get_model.side_effect = [Mock(), Exception("Not found")]
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.project_id = "test-project"
        trainer.dataset_id = "books"
        trainer.matrix_factorization_model = "test-project.books.matrix_factorization_model"
        trainer.boosted_tree_model = "test-project.books.boosted_tree_regressor_model"
        trainer.automl_regressor_model = "test-project.books.automl_regressor_model"
        
        trainer.check_and_cleanup_existing_models()
        
        # Should not raise exception
        assert True
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.time.sleep')
    def test_safe_train_model_success(self, mock_sleep, mock_client_class):
        """Test successful model training"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        
        result = trainer.safe_train_model(
            "test-model",
            "CREATE MODEL test-model AS SELECT 1",
            "TEST_MODEL"
        )
        
        assert result is True
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.time.sleep')
    def test_safe_train_model_retry(self, mock_sleep, mock_client_class):
        """Test model training with retry"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = [Exception("Error"), None]
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        
        result = trainer.safe_train_model(
            "test-model",
            "CREATE MODEL test-model AS SELECT 1",
            "TEST_MODEL",
            max_retries=3
        )
        
        assert result is True
        assert mock_sleep.called

