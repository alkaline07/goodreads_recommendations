"""
Comprehensive test cases for model_evaluation_pipeline.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import os
import json
from src.model_evaluation_pipeline import ModelEvaluationPipeline, safe_mlflow_log


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
    
    def test_safe_mlflow_log_with_multiple_args(self):
        """Test MLflow logging with multiple arguments"""
        mock_func = Mock(return_value="success")
        result = safe_mlflow_log(mock_func, "arg1", "arg2", kwarg1="value1")
        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")


class TestModelEvaluationPipeline:
    """Comprehensive test cases for ModelEvaluationPipeline class"""
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_analyzer_class, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        pipeline = ModelEvaluationPipeline()
        
        assert pipeline.project_id == "test-project"
        assert pipeline.dataset_id == "books"
        assert pipeline.sensitivity_analyzer is not None
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    def test_init_without_airflow_home(self, mock_analyzer_class, mock_client_class):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        with patch.dict('os.environ', {}, clear=True):
            pipeline = ModelEvaluationPipeline()
            assert pipeline.project_id == "test-project"
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    def test_init_with_custom_project_id(self, mock_analyzer_class, mock_client_class):
        """Test initialization with custom project_id"""
        mock_client = Mock()
        mock_client.project = "custom-project"
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        pipeline = ModelEvaluationPipeline(project_id="custom-project")
        assert pipeline.project_id == "custom-project"
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    @patch('src.model_evaluation_pipeline.mlflow.end_run')
    @patch('src.model_evaluation_pipeline.mlflow.start_run')
    def test_evaluate_model_success(self, mock_mlflow_run, mock_mlflow_end_run, mock_analyzer_class, mock_client_class):
        """Test evaluating model successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2],
            'correlation': [0.8],
            'r_squared': [0.64],
            'accuracy_within_0_5_pct': [60.0],
            'accuracy_within_1_0_pct': [80.0],
            'accuracy_within_1_5_pct': [90.0]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_mlflow_context = Mock()
        mock_mlflow_context.info.run_id = "test-run-id"
        mock_mlflow_run.return_value.__enter__.return_value = mock_mlflow_context
        mock_mlflow_run.return_value.__exit__.return_value = None
        mock_mlflow_end_run.return_value = None
        
        pipeline = ModelEvaluationPipeline()
        pipeline.client = mock_client
        pipeline.sensitivity_analyzer = mock_analyzer
        
        results = pipeline.evaluate_model(
            model_name="test-model",
            predictions_table="test-table",
            run_sensitivity_analysis=False
        )
        
        assert results['model_name'] == "test-model"
        assert results['performance_metrics'] is not None
        assert results['performance_metrics']['num_predictions'] == 1000
        assert results['performance_metrics']['mae'] == 0.5
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    @patch('src.model_evaluation_pipeline.mlflow.end_run')
    @patch('src.model_evaluation_pipeline.mlflow.start_run')
    def test_evaluate_model_with_sensitivity_analysis(self, mock_mlflow_run, mock_mlflow_end_run, mock_analyzer_class, mock_client_class):
        """Test evaluating model with sensitivity analysis"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2],
            'correlation': [0.8],
            'r_squared': [0.64],
            'accuracy_within_0_5_pct': [60.0],
            'accuracy_within_1_0_pct': [80.0],
            'accuracy_within_1_5_pct': [90.0]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer.analyze_feature_importance.return_value = {
            'model_name': 'test-model',
            'feature_importance': [
                {'feature': 'feature1', 'importance': 0.5},
                {'feature': 'feature2', 'importance': 0.3}
            ]
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_mlflow_context = Mock()
        mock_mlflow_context.info.run_id = "test-run-id"
        mock_mlflow_run.return_value.__enter__.return_value = mock_mlflow_context
        mock_mlflow_run.return_value.__exit__.return_value = None
        mock_mlflow_end_run.return_value = None
        
        pipeline = ModelEvaluationPipeline()
        pipeline.client = mock_client
        pipeline.sensitivity_analyzer = mock_analyzer
        
        results = pipeline.evaluate_model(
            model_name="test-model",
            predictions_table="test-table",
            run_sensitivity_analysis=True,
            sensitivity_sample_size=1000
        )
        
        assert results['model_name'] == "test-model"
        assert results['sensitivity_analysis'] is not None
        mock_analyzer.analyze_feature_importance.assert_called_once()
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    @patch('src.model_evaluation_pipeline.mlflow.end_run')
    @patch('src.model_evaluation_pipeline.mlflow.start_run')
    def test_evaluate_model_sensitivity_analysis_exception(self, mock_mlflow_run, mock_mlflow_end_run, mock_analyzer_class, mock_client_class):
        """Test exception handling in sensitivity analysis"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2],
            'correlation': [0.8],
            'r_squared': [0.64],
            'accuracy_within_0_5_pct': [60.0],
            'accuracy_within_1_0_pct': [80.0],
            'accuracy_within_1_5_pct': [90.0]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer.analyze_feature_importance.side_effect = Exception("Analysis failed")
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_mlflow_context = Mock()
        mock_mlflow_context.info.run_id = "test-run-id"
        mock_mlflow_run.return_value.__enter__.return_value = mock_mlflow_context
        mock_mlflow_run.return_value.__exit__.return_value = None
        mock_mlflow_end_run.return_value = None
        
        pipeline = ModelEvaluationPipeline()
        pipeline.client = mock_client
        pipeline.sensitivity_analyzer = mock_analyzer
        
        results = pipeline.evaluate_model(
            model_name="test-model",
            predictions_table="test-table",
            run_sensitivity_analysis=True
        )
        
        assert results['sensitivity_analysis'] is None
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    def test_compute_performance_metrics_success(self, mock_analyzer_class, mock_client_class):
        """Test computing performance metrics successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2],
            'correlation': [0.8],
            'r_squared': [0.64],
            'accuracy_within_0_5_pct': [60.0],
            'accuracy_within_1_0_pct': [80.0],
            'accuracy_within_1_5_pct': [90.0]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        pipeline = ModelEvaluationPipeline()
        pipeline.client = mock_client
        
        metrics = pipeline._compute_performance_metrics("test-table")
        
        assert metrics['num_predictions'] == 1000
        assert metrics['mae'] == 0.5
        assert metrics['rmse'] == 0.6
        assert metrics['r_squared'] == 0.64
        assert metrics['correlation'] == 0.8
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    def test_compute_performance_metrics_with_nan(self, mock_analyzer_class, mock_client_class):
        """Test computing performance metrics with NaN values"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [pd.NA],
            'correlation': [pd.NA],
            'r_squared': [pd.NA],
            'accuracy_within_0_5_pct': [60.0],
            'accuracy_within_1_0_pct': [80.0],
            'accuracy_within_1_5_pct': [90.0]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        pipeline = ModelEvaluationPipeline()
        pipeline.client = mock_client
        
        metrics = pipeline._compute_performance_metrics("test-table")
        
        assert metrics['std_error'] == 0.0
        assert metrics['correlation'] == 0.0
        assert metrics['r_squared'] == 0.0
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    def test_compute_performance_metrics_exception(self, mock_analyzer_class, mock_client_class):
        """Test exception handling in compute_performance_metrics"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        pipeline = ModelEvaluationPipeline()
        pipeline.client = mock_client
        
        metrics = pipeline._compute_performance_metrics("test-table")
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 0
    
    @patch('src.model_evaluation_pipeline.bigquery.Client')
    @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    def test_save_evaluation_report(self, mock_makedirs, mock_open, mock_analyzer_class, mock_client_class):
        """Test saving evaluation report"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        pipeline = ModelEvaluationPipeline()
        pipeline.client = mock_client
        
        evaluation_results = {
            'model_name': 'test-model',
            'timestamp': '2025-01-01',
            'performance_metrics': {'mae': 0.5},
            'sensitivity_analysis': None
        }
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        report_path = pipeline._save_evaluation_report(evaluation_results, "test-model")
        
        assert report_path is not None
        mock_makedirs.assert_called_once()
        mock_file.write.assert_called()
        # Collect all write calls and join them to reconstruct the JSON
        written_content = ''.join([call[0][0] for call in mock_file.write.call_args_list])
        written_data = json.loads(written_content)
        assert written_data['model_name'] == 'test-model'
