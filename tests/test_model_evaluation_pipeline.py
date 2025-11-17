# """
# Test cases for model_evaluation_pipeline.py
# """

# import pytest
# from unittest.mock import Mock, patch
# from src.model_evaluation_pipeline import ModelEvaluationPipeline, safe_mlflow_log


# class TestSafeMLflowLog:
#     """Test safe_mlflow_log function"""
    
#     def test_safe_mlflow_log_success(self):
#         """Test successful MLflow logging"""
#         mock_func = Mock(return_value="success")
#         result = safe_mlflow_log(mock_func, "arg1", kwarg1="value1")
#         assert result == "success"
    
#     def test_safe_mlflow_log_failure(self):
#         """Test MLflow logging with exception"""
#         mock_func = Mock(side_effect=Exception("MLflow error"))
#         result = safe_mlflow_log(mock_func, "arg1")
#         assert result is None


# class TestModelEvaluationPipeline:
#     """Test cases for ModelEvaluationPipeline class"""
    
#     @patch('src.model_evaluation_pipeline.bigquery.Client')
#     @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
#     def test_init(self, mock_analyzer_class, mock_client_class):
#         """Test initialization"""
#         mock_client = Mock()
#         mock_client.project = "test-project"
#         mock_client_class.return_value = mock_client
        
#         pipeline = ModelEvaluationPipeline()
        
#         assert pipeline.project_id == "test-project"
#         assert pipeline.dataset_id == "books"
    
#     @patch('src.model_evaluation_pipeline.bigquery.Client')
#     @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
#     @patch('src.model_evaluation_pipeline.mlflow.start_run')
#     def test_evaluate_model(self, mock_mlflow_run, mock_analyzer_class, mock_client_class):
#         """Test evaluating model"""
#         import pandas as pd
#         mock_client = Mock()
#         mock_client.project = "test-project"
#         mock_df = pd.DataFrame({
#             'num_predictions': [1000],
#             'mae': [0.5],
#             'rmse': [0.6],
#             'mean_predicted': [3.5],
#             'mean_actual': [3.6],
#             'std_error': [0.2],
#             'correlation': [0.8],
#             'r_squared': [0.64],
#             'accuracy_within_0_5_pct': [60.0],
#             'accuracy_within_1_0_pct': [80.0],
#             'accuracy_within_1_5_pct': [90.0]
#         })
#         mock_client.query.return_value.to_dataframe.return_value = mock_df
#         mock_client_class.return_value = mock_client
        
#         mock_analyzer = Mock()
#         mock_analyzer.analyze_feature_importance.return_value = {
#             'feature_importance': []
#         }
#         mock_analyzer_class.return_value = mock_analyzer
        
#         mock_mlflow_context = Mock()
#         mock_mlflow_run.return_value.__enter__.return_value = mock_mlflow_context
#         mock_mlflow_run.return_value.__exit__.return_value = None
        
#         pipeline = ModelEvaluationPipeline()
#         pipeline.client = mock_client
#         pipeline.sensitivity_analyzer = mock_analyzer
        
#         results = pipeline.evaluate_model(
#             model_name="test-model",
#             predictions_table="test-table",
#             run_sensitivity_analysis=False
#         )
        
#         assert results['model_name'] == "test-model"
#         assert results['performance_metrics'] is not None
    
#     @patch('src.model_evaluation_pipeline.bigquery.Client')
#     @patch('src.model_evaluation_pipeline.ModelSensitivityAnalyzer')
#     def test_compute_performance_metrics(self, mock_analyzer_class, mock_client_class):
#         """Test computing performance metrics"""
#         import pandas as pd
#         mock_client = Mock()
#         mock_client.project = "test-project"
#         mock_df = pd.DataFrame({
#             'num_predictions': [1000],
#             'mae': [0.5],
#             'rmse': [0.6],
#             'mean_predicted': [3.5],
#             'mean_actual': [3.6],
#             'std_error': [0.2],
#             'correlation': [0.8],
#             'r_squared': [0.64],
#             'accuracy_within_0_5_pct': [60.0],
#             'accuracy_within_1_0_pct': [80.0],
#             'accuracy_within_1_5_pct': [90.0]
#         })
#         mock_client.query.return_value.to_dataframe.return_value = mock_df
#         mock_client_class.return_value = mock_client
        
#         pipeline = ModelEvaluationPipeline()
#         pipeline.client = mock_client
        
#         metrics = pipeline._compute_performance_metrics("test-table")
        
#         assert metrics['num_predictions'] == 1000
#         assert metrics['mae'] == 0.5
#         assert metrics['rmse'] == 0.6

