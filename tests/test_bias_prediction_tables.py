"""
Comprehensive test cases for bias_prediction_tables.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
from src.bias_prediction_tables import BiasReadyPredictionGenerator


class TestBiasReadyPredictionGenerator:
    """Comprehensive test cases for BiasReadyPredictionGenerator class"""
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        
        assert generator.project_id == "test-project"
        assert generator.dataset_id == "books"
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_init_without_airflow_home(self, mock_client_class):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        with patch.dict('os.environ', {}, clear=True):
            generator = BiasReadyPredictionGenerator()
            assert generator.project_id == "test-project"
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_find_latest_model_success(self, mock_client_class):
        """Test finding latest model successfully"""
        mock_client = Mock()
        mock_model1 = Mock()
        mock_model1.model_id = "boosted_tree_regressor_model_20250101"
        mock_model1.created = datetime(2025, 1, 1)
        mock_model2 = Mock()
        mock_model2.model_id = "boosted_tree_regressor_model_20250102"
        mock_model2.created = datetime(2025, 1, 2)
        mock_client.list_models.return_value = [mock_model1, mock_model2]
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        model_path = generator.find_latest_model("boosted_tree_regressor_model")
        
        assert model_path is not None
        assert "boosted_tree_regressor_model_20250102" in model_path
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_find_latest_model_not_found(self, mock_client_class):
        """Test finding latest model when none exists"""
        mock_client = Mock()
        mock_client.list_models.return_value = []
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        model_path = generator.find_latest_model("nonexistent_model")
        
        assert model_path is None
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_find_latest_model_exception(self, mock_client_class):
        """Test exception handling in find_latest_model"""
        mock_client = Mock()
        mock_client.list_models.side_effect = Exception("List failed")
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        model_path = generator.find_latest_model("test_model")
        
        assert model_path is None
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_verify_test_table_exists_success(self, mock_client_class):
        """Test verifying test table exists successfully"""
        mock_client = Mock()
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        mock_df = pd.DataFrame({'cnt': [1000]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        result = generator.verify_test_table_exists()
        
        assert result is True
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_verify_test_table_exists_not_found(self, mock_client_class):
        """Test verifying test table when it doesn't exist"""
        mock_client = Mock()
        mock_client.get_table.side_effect = Exception("Table not found")
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        result = generator.verify_test_table_exists()
        
        assert result is False
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_verify_test_table_exists_zero_rows(self, mock_client_class):
        """Test verifying test table with zero rows"""
        mock_client = Mock()
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        mock_df = pd.DataFrame({'cnt': [0]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        result = generator.verify_test_table_exists()
        
        assert result is True  # Table exists, just empty
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_list_available_models_success(self, mock_client_class):
        """Test listing available models successfully"""
        mock_client = Mock()
        mock_model1 = Mock()
        mock_model1.model_id = "boosted_tree_regressor_model"
        mock_model2 = Mock()
        mock_model2.model_id = "matrix_factorization_model"
        mock_client.list_models.return_value = [mock_model1, mock_model2]
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.dataset_id = "books"
        
        models = generator.list_available_models()
        
        assert isinstance(models, dict)
        assert 'boosted_tree' in models or 'matrix_factorization' in models
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_list_available_models_empty(self, mock_client_class):
        """Test listing available models when none exist"""
        mock_client = Mock()
        mock_client.list_models.return_value = []
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.dataset_id = "books"
        
        models = generator.list_available_models()
        
        assert isinstance(models, dict)
        assert not any(models.values())
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_list_available_models_exception(self, mock_client_class):
        """Test exception handling in list_available_models"""
        mock_client = Mock()
        mock_client.list_models.side_effect = Exception("List failed")
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.dataset_id = "books"
        
        models = generator.list_available_models()
        
        assert isinstance(models, dict)
        assert len(models) == 0
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_generate_boosted_tree_predictions_success(self, mock_client_class):
        """Test generating boosted tree predictions successfully"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        mock_stats_df = pd.DataFrame({
            'num_predictions': [1000],
            'mean_absolute_error': [0.5],
            'root_mean_squared_error': [0.6],
            'avg_predicted_rating': [3.5],
            'avg_actual_rating': [3.6]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_stats_df
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        result = generator.generate_boosted_tree_predictions("test-model")
        
        assert result is True
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_generate_boosted_tree_predictions_exception(self, mock_client_class):
        """Test exception handling in generate_boosted_tree_predictions"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = Exception("Query failed")
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        result = generator.generate_boosted_tree_predictions("test-model")
        
        assert result is False
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_generate_matrix_factorization_predictions_success(self, mock_client_class):
        """Test generating matrix factorization predictions successfully"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        mock_stats_df = pd.DataFrame({
            'num_predictions': [1000],
            'mean_absolute_error': [0.5],
            'root_mean_squared_error': [0.6],
            'avg_predicted_rating': [3.5],
            'avg_actual_rating': [3.6]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_stats_df
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        result = generator.generate_matrix_factorization_predictions("test-model")
        
        assert result is True
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_generate_matrix_factorization_predictions_exception(self, mock_client_class):
        """Test exception handling in generate_matrix_factorization_predictions"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = Exception("Query failed")
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        result = generator.generate_matrix_factorization_predictions("test-model")
        
        assert result is False
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_run_with_all_models(self, mock_client_class):
        """Test running pipeline with all model types"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_model = Mock()
        mock_model.model_id = "boosted_tree_regressor_model"
        mock_model.created = datetime(2025, 1, 1)
        mock_client.list_models.return_value = [mock_model]
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        mock_df = pd.DataFrame({'cnt': [1000]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        with patch.object(generator, 'generate_boosted_tree_predictions', return_value=True):
            with patch.object(generator, 'generate_matrix_factorization_predictions', return_value=True):
                generator.run(model_types=['boosted_tree', 'matrix_factorization'])
                
                # Should not raise exception
                assert True
    
    @patch('src.bias_prediction_tables.bigquery.Client')
    def test_run_with_no_test_table(self, mock_client_class):
        """Test running pipeline when test table doesn't exist"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.get_table.side_effect = Exception("Table not found")
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        generator.run()
        
        # Should not raise exception, just return early
        assert True
