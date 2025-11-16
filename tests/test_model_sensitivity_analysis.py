"""
Test cases for model_sensitivity_analysis.py
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.model_sensitivity_analysis import ModelSensitivityAnalyzer


class TestModelSensitivityAnalyzer:
    """Test cases for ModelSensitivityAnalyzer class"""
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    @patch('os.makedirs')
    def test_init(self, mock_makedirs, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        assert analyzer.project_id == "test-project"
        assert analyzer.dataset_id == "books"
        mock_makedirs.assert_called()
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_load_model_data(self, mock_client_class):
        """Test loading model data"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'predicted_rating': [3.5, 4.0, 3.0],
            'actual_rating': [3.6, 4.1, 2.9],
            'book_popularity_normalized': [0.5, 0.6, 0.4]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        analyzer.client = mock_client
        
        result = analyzer.load_model_data("test-table", sample_size=100)
        
        assert len(result) == 3
        assert 'predicted_rating' in result.columns
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_prepare_features_for_shap(self, mock_client_class):
        """Test preparing features for SHAP"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        df = pd.DataFrame({
            'predicted_rating': [3.5],
            'actual_rating': [3.6],
            'book_era': ['Modern'],
            'book_popularity_normalized': [0.5]
        })
        
        X, feature_names, categorical_mappings = analyzer.prepare_features_for_shap(df)
        
        assert X.shape[0] == 1
        assert 'book_popularity_normalized' in feature_names
        assert 'predicted_rating' not in feature_names
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    @patch('src.model_sensitivity_analysis.shap.KernelExplainer')
    @patch('os.makedirs')
    def test_analyze_feature_importance(self, mock_makedirs, mock_shap, mock_client_class):
        """Test analyzing feature importance"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'predicted_rating': [3.5, 4.0],
            'actual_rating': [3.6, 4.1],
            'book_popularity_normalized': [0.5, 0.6],
            'num_genres': [2, 3]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.array([[0.1, 0.2], [0.15, 0.25]])
        mock_shap.return_value = mock_explainer
        
        analyzer = ModelSensitivityAnalyzer()
        analyzer.client = mock_client
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                results = analyzer.analyze_feature_importance(
                    "test-table",
                    "test-model",
                    sample_size=100
                )
                
                assert results['model_name'] == "test-model"
                assert 'feature_importance' in results

