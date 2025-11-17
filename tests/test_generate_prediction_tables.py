"""
Test cases for generate_bias_prediction_tables.py
"""

import pytest
from unittest.mock import Mock, patch
from src.generate_prediction_tables import BiasReadyPredictionGenerator


class TestBiasReadyPredictionGenerator:
    """Test cases for BiasReadyPredictionGenerator class"""
    
    @patch('src.generate_prediction_tables.bigquery.Client')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        
        assert generator.project_id == "test-project"
        assert generator.dataset_id == "books"
    
    @patch('src.generate_prediction_tables.bigquery.Client')
    def test_find_latest_model(self, mock_client_class):
        """Test finding latest model"""
        from datetime import datetime
        
        mock_client = Mock()
        mock_model = Mock()
        mock_model.model_id = "boosted_tree_regressor_model_20250101"
        mock_model.created = datetime(2025, 1, 1)
        mock_client.list_models.return_value = [mock_model]
        mock_client_class.return_value = mock_client
        
        generator = BiasReadyPredictionGenerator()
        generator.client = mock_client
        generator.project_id = "test-project"
        generator.dataset_id = "books"
        
        model_path = generator.find_latest_model("boosted_tree_regressor_model")
        
        assert model_path is not None
        assert "boosted_tree_regressor_model" in model_path
    
    @patch('src.generate_prediction_tables.bigquery.Client')
    def test_verify_test_table_exists(self, mock_client_class):
        """Test verifying test table exists"""
        import pandas as pd
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
    
    @patch('src.generate_prediction_tables.bigquery.Client')
    def test_list_available_models(self, mock_client_class):
        """Test listing available models"""
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

