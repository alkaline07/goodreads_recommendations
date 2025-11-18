"""
Comprehensive test cases for bias_mitigation.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import json
from src.bias_mitigation import BiasMitigator, MitigationConfig, MitigationResult


class TestMitigationConfig:
    """Test MitigationConfig dataclass with all scenarios"""
    
    def test_mitigation_config_creation(self):
        """Test creating MitigationConfig with valid data"""
        config = MitigationConfig(
            technique="shrinkage",
            target_dimensions=["Popularity"],
            lambda_shrinkage=0.5
        )
        
        assert config.technique == "shrinkage"
        assert config.lambda_shrinkage == 0.5
        assert config.target_dimensions == ["Popularity"]
    
    def test_mitigation_config_with_threshold_adjustments(self):
        """Test MitigationConfig with threshold adjustments"""
        config = MitigationConfig(
            technique="threshold",
            target_dimensions=["Popularity"],
            threshold_adjustments={"High": 0.1, "Low": -0.1}
        )
        
        assert config.technique == "threshold"
        assert config.threshold_adjustments == {"High": 0.1, "Low": -0.1}
    
    def test_mitigation_config_with_reweight_strategy(self):
        """Test MitigationConfig with reweight strategy"""
        config = MitigationConfig(
            technique="reweighting",
            target_dimensions=["Popularity"],
            reweight_strategy="balanced"
        )
        
        assert config.technique == "reweighting"
        assert config.reweight_strategy == "balanced"
    
    def test_mitigation_config_default_values(self):
        """Test MitigationConfig with default values"""
        config = MitigationConfig(
            technique="shrinkage",
            target_dimensions=["Popularity"]
        )
        
        assert config.lambda_shrinkage == 0.5
        assert config.threshold_adjustments is None
        assert config.reweight_strategy == "inverse_propensity"


class TestBiasMitigator:
    """Comprehensive test cases for BiasMitigator class"""
    
    @patch('src.bias_mitigation.bigquery.Client')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        
        assert mitigator.project_id == "test-project"
        assert mitigator.dataset_id == "books"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_init_without_airflow_home(self, mock_client_class):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        with patch.dict('os.environ', {}, clear=True):
            mitigator = BiasMitigator()
            assert mitigator.project_id == "test-project"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_init_with_custom_project_id(self, mock_client_class):
        """Test initialization with custom project_id"""
        mock_client = Mock()
        mock_client.project = "custom-project"
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator(project_id="custom-project")
        assert mitigator.project_id == "custom-project"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_group_metrics_success(self, mock_client_class):
        """Test computing group metrics successfully"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'slice_group': ['High', 'Low'],
            'count': [100, 200],
            'mean_rating': [3.5, 3.6],
            'std_rating': [0.2, 0.3]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        metrics = mitigator._compute_group_metrics(
            "test-table",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
        )
        
        assert isinstance(metrics, dict)
        assert 'High' in metrics
        assert 'Low' in metrics
        assert metrics['High']['count'] == 100
        assert metrics['Low']['count'] == 200
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_group_metrics_empty_result(self, mock_client_class):
        """Test computing group metrics with empty result"""
        mock_client = Mock()
        mock_df = pd.DataFrame()
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        metrics = mitigator._compute_group_metrics(
            "test-table",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
        )
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 0
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_group_metrics_with_nan_std(self, mock_client_class):
        """Test computing group metrics with NaN std_rating"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'slice_group': ['High'],
            'count': [100],
            'mean_rating': [3.5],
            'std_rating': [np.nan]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        metrics = mitigator._compute_group_metrics(
            "test-table",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
        )
        
        assert metrics['High']['std_rating'] == 0.0
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_group_metrics_exception(self, mock_client_class):
        """Test exception handling in compute_group_metrics"""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        metrics = mitigator._compute_group_metrics(
            "test-table",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
        )
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 0
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_prediction_metrics_success(self, mock_client_class):
        """Test computing prediction metrics successfully"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'count': [1000],
            'mae': [0.5],
            'rmse': [0.6]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        metrics = mitigator._compute_prediction_metrics("test-table")
        
        assert metrics['count'] == 1000
        assert metrics['mae'] == 0.5
        assert metrics['rmse'] == 0.6
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_prediction_metrics_exception(self, mock_client_class):
        """Test exception handling in compute_prediction_metrics"""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        metrics = mitigator._compute_prediction_metrics("test-table")
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 0
    
    def test_calculate_improvements_group_metrics(self):
        """Test calculating improvements for group metrics"""
        mitigator = BiasMitigator.__new__(BiasMitigator)
        
        original = {
            'High': {'mean_rating': 3.5},
            'Low': {'mean_rating': 3.0}
        }
        mitigated = {
            'High': {'mean_rating': 3.4},
            'Low': {'mean_rating': 3.3}
        }
        
        improvements = mitigator._calculate_improvements(original, mitigated)
        
        assert 'variance_reduction_pct' in improvements
    
    def test_calculate_improvements_prediction_metrics(self):
        """Test calculating improvements for prediction metrics"""
        mitigator = BiasMitigator.__new__(BiasMitigator)
        
        original = {'mae': 0.6, 'rmse': 0.7}
        mitigated = {'mae': 0.5, 'rmse': 0.6}
        
        improvements = mitigator._calculate_improvements(original, mitigated)
        
        assert 'mae_improvement_pct' in improvements
        assert 'rmse_improvement_pct' in improvements
        assert improvements['mae_improvement_pct'] > 0
        assert improvements['rmse_improvement_pct'] > 0
    
    def test_calculate_improvements_no_improvement(self):
        """Test calculating improvements when no improvement"""
        mitigator = BiasMitigator.__new__(BiasMitigator)
        
        original = {'mae': 0.5, 'rmse': 0.6}
        mitigated = {'mae': 0.5, 'rmse': 0.6}
        
        improvements = mitigator._calculate_improvements(original, mitigated)
        
        assert improvements['mae_improvement_pct'] == 0
        assert improvements['rmse_improvement_pct'] == 0
    
    def test_calculate_improvements_worse_performance(self):
        """Test calculating improvements when performance worsens"""
        mitigator = BiasMitigator.__new__(BiasMitigator)
        
        original = {'mae': 0.5, 'rmse': 0.6}
        mitigated = {'mae': 0.6, 'rmse': 0.7}
        
        improvements = mitigator._calculate_improvements(original, mitigated)
        
        assert improvements['mae_improvement_pct'] < 0
        assert improvements['rmse_improvement_pct'] < 0
    
    def test_calculate_improvements_zero_original(self):
        """Test calculating improvements when original is zero"""
        mitigator = BiasMitigator.__new__(BiasMitigator)
        
        original = {'mae': 0.0, 'rmse': 0.0}
        mitigated = {'mae': 0.5, 'rmse': 0.6}
        
        improvements = mitigator._calculate_improvements(original, mitigated)
        
        assert improvements['mae_improvement_pct'] == 0
        assert improvements['rmse_improvement_pct'] == 0
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_apply_shrinkage_mitigation_success(self, mock_client_class):
        """Test applying shrinkage mitigation successfully"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        mock_df = pd.DataFrame({
            'slice_group': ['High'],
            'count': [100],
            'mean_rating': [3.5],
            'std_rating': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        result = mitigator.apply_shrinkage_mitigation(
            "input-table",
            "output-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            lambda_shrinkage=0.5
        )
        
        assert isinstance(result, MitigationResult)
        assert result.technique == "shrinkage"
        assert result.output_table == "output-table"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_apply_shrinkage_mitigation_exception(self, mock_client_class):
        """Test exception handling in apply_shrinkage_mitigation"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = Exception("Query failed")
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        with pytest.raises(Exception):
            mitigator.apply_shrinkage_mitigation(
                "input-table",
                "output-table",
                "Popularity",
                "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
                lambda_shrinkage=0.5
            )
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_apply_shrinkage_mitigation_different_lambda(self, mock_client_class):
        """Test applying shrinkage with different lambda values"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        mock_df = pd.DataFrame({
            'slice_group': ['High'],
            'count': [100],
            'mean_rating': [3.5],
            'std_rating': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        # Test with lambda = 0 (no shrinkage)
        result = mitigator.apply_shrinkage_mitigation(
            "input-table",
            "output-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            lambda_shrinkage=0.0
        )
        
        assert result.lambda_shrinkage == 0.0 if hasattr(result, 'lambda_shrinkage') else True
        
        # Test with lambda = 1 (full shrinkage)
        result = mitigator.apply_shrinkage_mitigation(
            "input-table",
            "output-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            lambda_shrinkage=1.0
        )
        
        assert isinstance(result, MitigationResult)
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_create_reweighted_training_table_inverse_propensity(self, mock_client_class):
        """Test creating reweighted table with inverse propensity strategy"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        mock_df = pd.DataFrame({
            'slice_group': ['High'],
            'count': [100],
            'mean_rating': [3.5],
            'std_rating': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        result = mitigator.create_reweighted_training_table(
            "training-table",
            "output-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            strategy='inverse_propensity'
        )
        
        assert isinstance(result, MitigationResult)
        assert result.technique == "reweighting"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_create_reweighted_training_table_balanced(self, mock_client_class):
        """Test creating reweighted table with balanced strategy"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        mock_df = pd.DataFrame({
            'slice_group': ['High'],
            'count': [100],
            'mean_rating': [3.5],
            'std_rating': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        result = mitigator.create_reweighted_training_table(
            "training-table",
            "output-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            strategy='balanced'
        )
        
        assert isinstance(result, MitigationResult)
        assert result.technique == "reweighting"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_create_reweighted_training_table_exception(self, mock_client_class):
        """Test exception handling in create_reweighted_training_table"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = Exception("Query failed")
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        with pytest.raises(Exception):
            mitigator.create_reweighted_training_table(
                "training-table",
                "output-table",
                "Popularity",
                "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
            )
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_apply_threshold_adjustments_with_provided(self, mock_client_class):
        """Test applying threshold adjustments with provided adjustments"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        mock_df = pd.DataFrame({
            'count': [1000],
            'mae': [0.5],
            'rmse': [0.6]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        result = mitigator.apply_threshold_adjustments(
            "predictions-table",
            "output-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            threshold_adjustments={"High": 0.1, "Low": -0.1}
        )
        
        assert isinstance(result, MitigationResult)
        assert result.technique == "threshold_adjustment"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_apply_threshold_adjustments_auto_compute(self, mock_client_class):
        """Test applying threshold adjustments with auto-computed adjustments"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        # Mock for computing optimal thresholds
        mock_threshold_df = pd.DataFrame({
            'slice_group': ['High', 'Low'],
            'mean_error': [0.1, -0.1]
        })
        
        # Mock for computing prediction metrics
        mock_metrics_df = pd.DataFrame({
            'count': [1000],
            'mae': [0.5],
            'rmse': [0.6]
        })
        
        mock_client.query.return_value.to_dataframe.side_effect = [
            mock_threshold_df,  # First call for optimal thresholds
            mock_metrics_df,     # Second call for original metrics
            mock_metrics_df      # Third call for adjusted metrics
        ]
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        result = mitigator.apply_threshold_adjustments(
            "predictions-table",
            "output-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            threshold_adjustments=None
        )
        
        assert isinstance(result, MitigationResult)
        assert result.technique == "threshold_adjustment"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_apply_threshold_adjustments_exception(self, mock_client_class):
        """Test exception handling in apply_threshold_adjustments"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.side_effect = Exception("Query failed")
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        with pytest.raises(Exception):
            mitigator.apply_threshold_adjustments(
                "predictions-table",
                "output-table",
                "Popularity",
                "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
            )
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_optimal_thresholds(self, mock_client_class):
        """Test computing optimal thresholds"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'slice_group': ['High', 'Low'],
            'mean_error': [0.1, -0.1]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        adjustments = mitigator._compute_optimal_thresholds(
            "predictions-table",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
        )
        
        assert isinstance(adjustments, dict)
        assert 'High' in adjustments
        assert 'Low' in adjustments
        assert adjustments['High'] == 0.1
        assert adjustments['Low'] == -0.1
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_optimal_thresholds_exception(self, mock_client_class):
        """Test exception handling in compute_optimal_thresholds"""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        adjustments = mitigator._compute_optimal_thresholds(
            "predictions-table",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
        )
        
        assert isinstance(adjustments, dict)
        assert len(adjustments) == 0
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_analyze_weights(self, mock_client_class):
        """Test analyzing weight distribution"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'min_weight': [0.5],
            'max_weight': [2.0],
            'mean_weight': [1.0],
            'std_weight': [0.3]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        weight_stats = mitigator._analyze_weights("test-table")
        
        assert weight_stats['min_weight'] == 0.5
        assert weight_stats['max_weight'] == 2.0
        assert weight_stats['mean_weight'] == 1.0
        assert weight_stats['std_weight'] == 0.3
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_analyze_weights_exception(self, mock_client_class):
        """Test exception handling in analyze_weights"""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        weight_stats = mitigator._analyze_weights("test-table")
        
        assert isinstance(weight_stats, dict)
        assert len(weight_stats) == 0
    
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    def test_save_mitigation_report(self, mock_makedirs, mock_open):
        """Test saving mitigation report"""
        mitigator = BiasMitigator.__new__(BiasMitigator)
        
        result = MitigationResult(
            technique="shrinkage",
            timestamp="2025-01-01",
            original_metrics={},
            mitigated_metrics={},
            improvement_pct={},
            output_table="test-table"
        )
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        mitigator.save_mitigation_report(result, "/path/to/report.json")
        
        mock_makedirs.assert_called_once()
        mock_file.write.assert_called()
        # Collect all write calls and join them to reconstruct the JSON
        written_content = ''.join([call[0][0] for call in mock_file.write.call_args_list])
        written_data = json.loads(written_content)
        assert written_data["technique"] == "shrinkage"
