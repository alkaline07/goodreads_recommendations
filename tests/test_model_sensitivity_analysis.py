"""
Comprehensive test cases for model_sensitivity_analysis.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from src.model_sensitivity_analysis import ModelSensitivityAnalyzer


class TestModelSensitivityAnalyzer:
    """Comprehensive test cases for ModelSensitivityAnalyzer class"""
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    @patch('os.makedirs')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_makedirs, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        assert analyzer.project_id == "test-project"
        assert analyzer.dataset_id == "books"
        assert analyzer.output_dir == "../docs/model_analysis/sensitivity"
        mock_makedirs.assert_called()
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    @patch('os.makedirs')
    def test_init_without_airflow_home(self, mock_makedirs, mock_client_class):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        with patch.dict('os.environ', {}, clear=True):
            analyzer = ModelSensitivityAnalyzer()
            assert analyzer.project_id == "test-project"
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    @patch('os.makedirs')
    def test_init_with_custom_project_id(self, mock_makedirs, mock_client_class):
        """Test initialization with custom project_id"""
        mock_client = Mock()
        mock_client.project = "custom-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer(project_id="custom-project")
        assert analyzer.project_id == "custom-project"
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_load_model_data_success(self, mock_client_class):
        """Test loading model data successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'predicted_rating': [3.5, 4.0, 3.0],
            'actual_rating': [3.6, 4.1, 2.9],
            'book_popularity_normalized': [0.5, 0.6, 0.4],
            'num_genres': [2, 3, 1]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        analyzer.client = mock_client
        
        result = analyzer.load_model_data("test-table", sample_size=100)
        
        assert len(result) == 3
        assert 'predicted_rating' in result.columns
        assert 'actual_rating' in result.columns
        assert 'book_popularity_normalized' in result.columns
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_load_model_data_empty_result(self, mock_client_class):
        """Test loading model data with empty result"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame()
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        analyzer.client = mock_client
        
        result = analyzer.load_model_data("test-table", sample_size=100)
        
        assert len(result) == 0
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_load_model_data_exception(self, mock_client_class):
        """Test exception handling in load_model_data"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        analyzer.client = mock_client
        
        with pytest.raises(Exception):
            analyzer.load_model_data("test-table", sample_size=100)
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_prepare_features_for_shap_success(self, mock_client_class):
        """Test preparing features for SHAP successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        df = pd.DataFrame({
            'predicted_rating': [3.5, 4.0],
            'actual_rating': [3.6, 4.1],
            'book_era': ['Modern', 'Classic'],
            'book_popularity_normalized': [0.5, 0.6],
            'num_genres': [2, 3]
        })
        
        X, feature_names, categorical_mappings = analyzer.prepare_features_for_shap(df)
        
        assert X.shape[0] == 2
        assert 'book_popularity_normalized' in feature_names
        assert 'num_genres' in feature_names
        assert 'predicted_rating' not in feature_names
        assert 'actual_rating' not in feature_names
        assert 'book_era' in categorical_mappings
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_prepare_features_for_shap_with_nan(self, mock_client_class):
        """Test preparing features for SHAP with NaN values"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        df = pd.DataFrame({
            'predicted_rating': [3.5, np.nan],
            'actual_rating': [3.6, 4.1],
            'book_popularity_normalized': [0.5, np.nan],
            'num_genres': [2, 3]
        })
        
        X, feature_names, categorical_mappings = analyzer.prepare_features_for_shap(df)
        
        assert X.shape[0] == 2
        # NaN values should be filled with 0
        assert not np.isnan(X).any()
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    @patch('src.model_sensitivity_analysis.shap.KernelExplainer')
    @patch('os.makedirs')
    def test_analyze_feature_importance_success(self, mock_makedirs, mock_shap, mock_client_class):
        """Test analyzing feature importance successfully"""
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
                with patch('matplotlib.pyplot.figure'):
                    with patch('seaborn.heatmap'):
                        with patch('builtins.open', create=True):
                            results = analyzer.analyze_feature_importance(
                                "test-table",
                                "test-model",
                                sample_size=100
                            )
                            
                            assert results['model_name'] == "test-model"
                            assert 'feature_importance' in results
                            assert isinstance(results['feature_importance'], list)
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    @patch('src.model_sensitivity_analysis.shap.KernelExplainer')
    @patch('os.makedirs')
    def test_analyze_feature_importance_exception(self, mock_makedirs, mock_shap, mock_client_class):
        """Test exception handling in analyze_feature_importance"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'predicted_rating': [3.5, 4.0],
            'actual_rating': [3.6, 4.1],
            'book_popularity_normalized': [0.5, 0.6]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_shap.side_effect = Exception("SHAP failed")
        
        analyzer = ModelSensitivityAnalyzer()
        analyzer.client = mock_client
        
        with pytest.raises(Exception):
            analyzer.analyze_feature_importance(
                "test-table",
                "test-model",
                sample_size=100
            )
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_prepare_features_for_shap_all_categorical(self, mock_client_class):
        """Test preparing features when all are categorical"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        df = pd.DataFrame({
            'predicted_rating': [3.5],
            'actual_rating': [3.6],
            'book_era': ['Modern'],
            'book_length_category': ['Short'],
            'reading_pace_category': ['Fast'],
            'author_gender_group': ['Male']
        })
        
        X, feature_names, categorical_mappings = analyzer.prepare_features_for_shap(df)
        
        assert len(categorical_mappings) == 4
        assert 'book_era' in categorical_mappings
        assert 'book_length_category' in categorical_mappings
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_prepare_features_for_shap_all_numeric(self, mock_client_class):
        """Test preparing features when all are numeric"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        df = pd.DataFrame({
            'predicted_rating': [3.5],
            'actual_rating': [3.6],
            'book_popularity_normalized': [0.5],
            'num_genres': [2],
            'user_activity_count': [10]
        })
        
        X, feature_names, categorical_mappings = analyzer.prepare_features_for_shap(df)
        
        assert len(categorical_mappings) == 0
        assert len(feature_names) == 3
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_compare_model_sensitivities(self, mock_client_class):
        """Test comparing model sensitivities"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        model_results = {
            'model1': {
                'feature_importance': [
                    {'feature': 'feature1', 'importance': 0.5},
                    {'feature': 'feature2', 'importance': 0.3}
                ]
            },
            'model2': {
                'feature_importance': [
                    {'feature': 'feature1', 'importance': 0.4},
                    {'feature': 'feature2', 'importance': 0.4}
                ]
            }
        }
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = [mock_ax]
        mock_fig.axes = mock_axes
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                with patch('matplotlib.pyplot.figure', return_value=mock_fig):
                    with patch('pandas.DataFrame.plot') as mock_plot:
                        mock_plot.return_value = mock_ax
                        comparison_df = analyzer.compare_model_sensitivities(model_results)
                        
                        assert isinstance(comparison_df, pd.DataFrame)
                        assert 'feature1' in comparison_df.index
                        assert 'feature2' in comparison_df.index
    
    @patch('src.model_sensitivity_analysis.bigquery.Client')
    def test_compare_model_sensitivities_single_model(self, mock_client_class):
        """Test comparing model sensitivities with single model"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        analyzer = ModelSensitivityAnalyzer()
        
        model_results = {
            'model1': {
                'feature_importance': [
                    {'feature': 'feature1', 'importance': 0.5}
                ]
            }
        }
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = [mock_ax]
        mock_fig.axes = mock_axes
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                with patch('matplotlib.pyplot.figure', return_value=mock_fig):
                    with patch('pandas.DataFrame.plot') as mock_plot:
                        mock_plot.return_value = mock_ax
                        comparison_df = analyzer.compare_model_sensitivities(model_results)
                        
                        assert isinstance(comparison_df, pd.DataFrame)
