"""
Comprehensive test cases for bq_model_training.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import time
from datetime import datetime
import pandas as pd
from src.bq_model_training import BigQueryMLModelTraining, safe_mlflow_log


class TestSafeMLflowLog:
    """Comprehensive test cases for safe_mlflow_log function"""
    
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

    def test_safe_mlflow_log_with_no_args(self):
        """Test MLflow logging with no arguments"""
        mock_func = Mock(return_value="success")
        result = safe_mlflow_log(mock_func)
        assert result == "success"
        mock_func.assert_called_once()


class TestBigQueryMLModelTraining:
    """Comprehensive test cases for BigQueryMLModelTraining class"""
    
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
        assert trainer.val_table == "test-project.books.goodreads_validation_set"
        assert trainer.test_table == "test-project.books.goodreads_test_set"
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_init_without_airflow_home(self, mock_client_class):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        with patch.dict('os.environ', {}, clear=True):
            trainer = BigQueryMLModelTraining()
            assert trainer.project_id == "test-project"
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_get_feature_columns_success(self, mock_client_class):
        """Test getting feature columns successfully"""
        mock_client = Mock()
        mock_table = Mock()
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
        assert 'book_id' not in features
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
    def test_get_feature_columns_empty_schema(self, mock_client_class):
        """Test get_feature_columns with empty schema"""
        mock_client = Mock()
        mock_table = Mock()
        mock_table.schema = []
        mock_client.get_table.return_value = mock_table
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        
        features = trainer.get_feature_columns()
        
        # Empty schema results in empty feature list (no exception, so no default)
        assert isinstance(features, list)
        assert len(features) == 0
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.time.sleep')
    def test_check_and_cleanup_existing_models_no_models(self, mock_sleep, mock_client_class):
        """Test checking existing models when none exist"""
        mock_client = Mock()
        mock_client.query.return_value.to_dataframe.return_value = Mock(empty=True)
        mock_client.get_model.side_effect = Exception("Not found")
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.project_id = "test-project"
        trainer.dataset_id = "books"
        
        trainer.check_and_cleanup_existing_models()
        
        # Should not raise exception
        assert True
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.time.sleep')
    def test_check_and_cleanup_existing_models_with_models(self, mock_sleep, mock_client_class):
        """Test checking existing models when models exist"""
        mock_client = Mock()
        mock_client.query.return_value.to_dataframe.return_value = Mock(empty=False)
        mock_client.get_model.return_value = Mock()
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
    def test_check_and_cleanup_existing_models_exception(self, mock_sleep, mock_client_class):
        """Test exception handling in check_and_cleanup_existing_models"""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        
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
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.time.sleep')
    def test_safe_train_model_max_retries_exceeded(self, mock_sleep, mock_client_class):
        """Test model training when max retries exceeded"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = Exception("Error")
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        
        result = trainer.safe_train_model(
            "test-model",
            "CREATE MODEL test-model AS SELECT 1",
            "TEST_MODEL",
            max_retries=2
        )
        
        assert result is False
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.time.sleep')
    def test_safe_train_model_concurrent_job(self, mock_sleep, mock_client_class):
        """Test model training with concurrent job error"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = Exception("multiple create model query jobs")
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        
        result = trainer.safe_train_model(
            "test-model",
            "CREATE MODEL test-model AS SELECT 1",
            "TEST_MODEL",
            max_retries=2
        )
        
        assert result is False
        assert mock_sleep.called
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.time.sleep')
    def test_safe_train_model_already_exists(self, mock_sleep, mock_client_class):
        """Test model training when model already exists"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = Exception("already exists")
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        
        result = trainer.safe_train_model(
            "test-model",
            "CREATE MODEL test-model AS SELECT 1",
            "TEST_MODEL",
            max_retries=2
        )
        
        # Should retry with timestamp suffix
        assert result is False
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.mlflow')
    def test_train_matrix_factorization_success(self, mock_mlflow, mock_client_class):
        """Test training matrix factorization model successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        trainer.matrix_factorization_model = "test-project.books.matrix_factorization_model"
        
        with patch.object(trainer, 'safe_train_model', return_value=True):
            with patch.object(trainer, 'evaluate_model'):
                result = trainer.train_matrix_factorization()
                assert result is True
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.mlflow')
    def test_train_matrix_factorization_failure(self, mock_mlflow, mock_client_class):
        """Test training matrix factorization model failure"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        trainer.matrix_factorization_model = "test-project.books.matrix_factorization_model"
        
        with patch.object(trainer, 'safe_train_model', return_value=False):
            result = trainer.train_matrix_factorization()
            assert result is False
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.mlflow')
    def test_train_boosted_tree_regressor_success(self, mock_mlflow, mock_client_class):
        """Test training boosted tree regressor successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        trainer.boosted_tree_model = "test-project.books.boosted_tree_regressor_model"
        
        with patch.object(trainer, 'get_feature_columns', return_value=['feature1', 'feature2']):
            with patch.object(trainer, 'safe_train_model', return_value=True):
                with patch.object(trainer, 'evaluate_model'):
                    result = trainer.train_boosted_tree_regressor()
                    assert result is True
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.mlflow')
    def test_train_boosted_tree_regressor_failure(self, mock_mlflow, mock_client_class):
        """Test training boosted tree regressor failure"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        trainer.boosted_tree_model = "test-project.books.boosted_tree_regressor_model"
        
        with patch.object(trainer, 'get_feature_columns', return_value=['feature1', 'feature2']):
            with patch.object(trainer, 'safe_train_model', return_value=False):
                result = trainer.train_boosted_tree_regressor()
                assert result is False
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_evaluate_model_success(self, mock_client_class):
        """Test evaluating model successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'mean_absolute_error': [0.5],
            'rmse': [0.6],
            'r2_score': [0.8]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.val_table = "test-project.books.goodreads_validation_set"
        
        trainer.evaluate_model("test-model", "TEST_MODEL")
        
        # Should not raise exception
        assert True
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_evaluate_model_empty_result(self, mock_client_class):
        """Test evaluating model with empty result"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame()
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.val_table = "test-project.books.goodreads_validation_set"
        
        trainer.evaluate_model("test-model", "TEST_MODEL")
        
        # Should not raise exception
        assert True
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_evaluate_model_exception(self, mock_client_class):
        """Test exception handling in evaluate_model"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.val_table = "test-project.books.goodreads_validation_set"
        
        trainer.evaluate_model("test-model", "TEST_MODEL")
        
        # Should not raise exception
        assert True
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_analyze_data_characteristics_success(self, mock_client_class):
        """Test analyzing data characteristics successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_stats_df = pd.DataFrame({
            'num_users': [1000],
            'num_books': [500],
            'num_interactions': [5000],
            'avg_rating': [3.5],
            'std_rating': [0.8],
            'min_rating': [1.0],
            'max_rating': [5.0]
        })
        mock_cold_start_df = pd.DataFrame({
            'users_with_few_ratings': [200],
            'total_users': [1000],
            'cold_start_ratio': [0.2]
        })
        mock_client.query.return_value.to_dataframe.side_effect = [mock_stats_df, mock_cold_start_df]
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        
        trainer.analyze_data_characteristics()
        
        assert hasattr(trainer, 'data_stats')
        assert trainer.data_stats['num_users'] == 1000
    
    @patch('src.bq_model_training.bigquery.Client')
    def test_analyze_data_characteristics_exception(self, mock_client_class):
        """Test exception handling in analyze_data_characteristics"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        trainer.train_table = "test-project.books.goodreads_train_set"
        
        trainer.analyze_data_characteristics()
        
        assert hasattr(trainer, 'data_stats')
        assert trainer.data_stats == {}
    
    @patch('src.bq_model_training.bigquery.Client')
    @patch('src.bq_model_training.mlflow')
    def test_run_pipeline(self, mock_mlflow, mock_client_class):
        """Test running the complete training pipeline"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        trainer = BigQueryMLModelTraining()
        trainer.client = mock_client
        
        with patch.object(trainer, 'check_and_cleanup_existing_models'):
            with patch.object(trainer, 'analyze_data_characteristics'):
                with patch.object(trainer, 'train_matrix_factorization', return_value=True):
                    with patch.object(trainer, 'train_boosted_tree_regressor', return_value=True):
                        trainer.run()
                        
                        # Should not raise exception
                        assert True
