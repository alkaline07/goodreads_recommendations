"""
Tests for Model Monitoring Module

Tests cover:
1. PSI calculation
2. Decay detection thresholds
3. Drift detection logic
4. Metric computation

Author: Goodreads Recommendation Team
Date: 2025
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime


class TestPSICalculation:
    """Tests for Population Stability Index calculation."""
    
    def test_psi_identical_distributions(self):
        """PSI should be ~0 for identical distributions."""
        from src.model_monitoring import ModelMonitor
        
        np.random.seed(42)
        baseline = np.random.normal(3.5, 1.0, 1000)
        current = np.random.normal(3.5, 1.0, 1000)
        
        with patch.object(ModelMonitor, '__init__', lambda x, project_id=None: None):
            monitor = ModelMonitor()
            psi = monitor._calculate_psi(baseline, current)
            
            assert psi < 0.1, f"PSI should be < 0.1 for similar distributions, got {psi}"
    
    def test_psi_shifted_distribution(self):
        """PSI should be high for significantly shifted distributions."""
        from src.model_monitoring import ModelMonitor
        
        np.random.seed(42)
        baseline = np.random.normal(3.0, 0.5, 1000)
        current = np.random.normal(4.5, 0.5, 1000)
        
        with patch.object(ModelMonitor, '__init__', lambda x, project_id=None: None):
            monitor = ModelMonitor()
            psi = monitor._calculate_psi(baseline, current)
            
            assert psi > 0.2, f"PSI should be > 0.2 for shifted distributions, got {psi}"
    
    def test_psi_moderate_shift(self):
        """PSI should be moderate for slight distribution shift."""
        from src.model_monitoring import ModelMonitor
        
        np.random.seed(42)
        baseline = np.random.normal(3.5, 1.0, 1000)
        current = np.random.normal(3.8, 1.1, 1000)
        
        with patch.object(ModelMonitor, '__init__', lambda x, project_id=None: None):
            monitor = ModelMonitor()
            psi = monitor._calculate_psi(baseline, current)
            
            assert 0.0 <= psi <= 0.3, f"PSI should be in moderate range, got {psi}"


class TestDecayThresholds:
    """Tests for model decay detection thresholds."""
    
    def test_decay_thresholds_exist(self):
        """Verify all required decay thresholds are defined."""
        from src.model_monitoring import ModelMonitor
        
        required_thresholds = [
            'rmse_increase_pct',
            'mae_increase_pct',
            'r_squared_decrease',
            'accuracy_decrease_pct'
        ]
        
        for threshold in required_thresholds:
            assert threshold in ModelMonitor.DECAY_THRESHOLDS, \
                f"Missing threshold: {threshold}"
    
    def test_drift_thresholds_exist(self):
        """Verify all required drift thresholds are defined."""
        from src.model_monitoring import ModelMonitor
        
        required_thresholds = [
            'ks_test_pvalue',
            'psi_threshold',
            'mean_shift_std'
        ]
        
        for threshold in required_thresholds:
            assert threshold in ModelMonitor.DRIFT_THRESHOLDS, \
                f"Missing threshold: {threshold}"
    
    def test_threshold_values_reasonable(self):
        """Verify threshold values are within reasonable ranges."""
        from src.model_monitoring import ModelMonitor
        
        assert 0 < ModelMonitor.DECAY_THRESHOLDS['rmse_increase_pct'] <= 20
        assert 0 < ModelMonitor.DECAY_THRESHOLDS['r_squared_decrease'] <= 0.2
        assert 0 < ModelMonitor.DRIFT_THRESHOLDS['psi_threshold'] <= 0.5
        assert 0 < ModelMonitor.DRIFT_THRESHOLDS['ks_test_pvalue'] <= 0.1


class TestMetricComputation:
    """Tests for metric computation logic."""
    
    @patch('src.model_monitoring.bigquery.Client')
    def test_compute_metrics_returns_expected_keys(self, mock_client):
        """Verify compute_metrics returns all expected metric keys."""
        from src.model_monitoring import ModelMonitor
        
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.65],
            'rmse': [0.85],
            'correlation': [0.7],
            'r_squared': [0.49],
            'accuracy_within_0_5': [45.0],
            'accuracy_within_1_0': [75.0]
        })
        
        mock_query = MagicMock()
        mock_query.to_dataframe.return_value = mock_df
        mock_client.return_value.query.return_value = mock_query
        mock_client.return_value.project = "test-project"
        
        with patch.object(ModelMonitor, '_ensure_metrics_tables_exist'):
            monitor = ModelMonitor(project_id="test-project")
            metrics = monitor._compute_metrics("test.table")
        
        expected_keys = [
            'num_predictions', 'mae', 'rmse', 'r_squared',
            'correlation', 'accuracy_within_0_5', 'accuracy_within_1_0'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: {key}"
    
    @patch('src.model_monitoring.bigquery.Client')
    def test_compute_metrics_handles_empty_result(self, mock_client):
        """Verify compute_metrics handles empty dataframe gracefully."""
        from src.model_monitoring import ModelMonitor
        
        mock_df = pd.DataFrame()
        
        mock_query = MagicMock()
        mock_query.to_dataframe.return_value = mock_df
        mock_client.return_value.query.return_value = mock_query
        mock_client.return_value.project = "test-project"
        
        with patch.object(ModelMonitor, '_ensure_metrics_tables_exist'):
            monitor = ModelMonitor(project_id="test-project")
            metrics = monitor._compute_metrics("test.table")
        
        assert metrics is None, "Should return None for empty results"


class TestDecayDetection:
    """Tests for model decay detection logic."""
    
    @patch('src.model_monitoring.bigquery.Client')
    def test_decay_detection_insufficient_data(self, mock_client):
        """Verify decay detection handles insufficient data."""
        from src.model_monitoring import ModelMonitor
        
        mock_df = pd.DataFrame()
        
        mock_query = MagicMock()
        mock_query.to_dataframe.return_value = mock_df
        mock_client.return_value.query.return_value = mock_query
        mock_client.return_value.project = "test-project"
        
        with patch.object(ModelMonitor, '_ensure_metrics_tables_exist'):
            monitor = ModelMonitor(project_id="test-project")
            
            with patch.object(monitor, '_save_decay_report'):
                result = monitor.detect_model_decay("test_model")
        
        assert result['decay_detected'] == False
        assert result.get('reason') == 'insufficient_data'
    
    @patch('src.model_monitoring.bigquery.Client')
    def test_decay_detection_rmse_increase(self, mock_client):
        """Verify decay is detected when RMSE increases significantly."""
        from src.model_monitoring import ModelMonitor
        
        baseline_df = pd.DataFrame({
            'metric_name': ['rmse', 'r_squared'],
            'avg_value': [0.8, 0.5],
            'std_value': [0.1, 0.05]
        })
        
        recent_df = pd.DataFrame({
            'metric_name': ['rmse', 'r_squared'],
            'avg_value': [1.0, 0.48],
            'std_value': [0.12, 0.06]
        })
        
        mock_query_baseline = MagicMock()
        mock_query_baseline.to_dataframe.return_value = baseline_df
        
        mock_query_recent = MagicMock()
        mock_query_recent.to_dataframe.return_value = recent_df
        
        mock_client.return_value.query.side_effect = [
            mock_query_baseline, mock_query_recent
        ]
        mock_client.return_value.project = "test-project"
        
        with patch.object(ModelMonitor, '_ensure_metrics_tables_exist'):
            monitor = ModelMonitor(project_id="test-project")
            
            with patch.object(monitor, '_save_decay_report'):
                result = monitor.detect_model_decay("test_model")
        
        assert result['decay_detected'] == True
        assert 'rmse' in result['metrics_comparison']


class TestDriftDetection:
    """Tests for data drift detection logic."""
    
    def test_ks_test_detects_different_distributions(self):
        """Verify KS test detects significantly different distributions."""
        from scipy import stats
        
        np.random.seed(42)
        baseline = np.random.normal(3.5, 1.0, 1000)
        current = np.random.normal(4.5, 1.0, 1000)
        
        ks_stat, ks_pvalue = stats.ks_2samp(baseline, current)
        
        assert ks_pvalue < 0.05, "KS test should detect different distributions"
    
    def test_ks_test_accepts_similar_distributions(self):
        """Verify KS test accepts similar distributions."""
        from scipy import stats
        
        np.random.seed(42)
        baseline = np.random.normal(3.5, 1.0, 1000)
        current = np.random.normal(3.5, 1.0, 1000)
        
        ks_stat, ks_pvalue = stats.ks_2samp(baseline, current)
        
        assert ks_pvalue > 0.05, "KS test should accept similar distributions"


class TestMonitorInitialization:
    """Tests for ModelMonitor initialization."""
    
    @patch('src.model_monitoring.bigquery.Client')
    @patch('src.model_monitoring.mlflow')
    def test_monitor_initializes_with_project_id(self, mock_mlflow, mock_client):
        """Verify monitor initializes correctly with project ID."""
        from src.model_monitoring import ModelMonitor
        
        mock_client.return_value.project = "test-project"
        
        with patch.object(ModelMonitor, '_ensure_metrics_tables_exist'):
            monitor = ModelMonitor(project_id="test-project")
        
        assert monitor.project_id == "test-project"
        assert monitor.dataset_id == "books"
    
    @patch('src.model_monitoring.bigquery.Client')
    @patch('src.model_monitoring.mlflow')
    def test_monitor_sets_correct_table_names(self, mock_mlflow, mock_client):
        """Verify monitor sets correct BigQuery table names."""
        from src.model_monitoring import ModelMonitor
        
        mock_client.return_value.project = "test-project"
        
        with patch.object(ModelMonitor, '_ensure_metrics_tables_exist'):
            monitor = ModelMonitor(project_id="test-project")
        
        assert "model_metrics_history" in monitor.metrics_table
        assert "data_drift_history" in monitor.drift_table


class TestAlertCreation:
    """Tests for alert creation logic."""
    
    @patch('src.model_monitoring.bigquery.Client')
    @patch('src.model_monitoring.mlflow')
    def test_alert_created_for_decay(self, mock_mlflow, mock_client):
        """Verify alert is created when decay is detected."""
        from src.model_monitoring import ModelMonitor
        
        mock_client.return_value.project = "test-project"
        
        with patch.object(ModelMonitor, '_ensure_metrics_tables_exist'):
            monitor = ModelMonitor(project_id="test-project")
        
        decay_results = {
            'decay_detected': True,
            'alerts': ['RMSE increased by 15%']
        }
        
        with patch.object(monitor, '_send_alert_notification') as mock_alert:
            result = monitor.create_alert_if_needed(decay_results=decay_results)
        
        assert result == True
        mock_alert.assert_called_once()
    
    @patch('src.model_monitoring.bigquery.Client')
    @patch('src.model_monitoring.mlflow')
    def test_no_alert_when_no_issues(self, mock_mlflow, mock_client):
        """Verify no alert is created when no issues detected."""
        from src.model_monitoring import ModelMonitor
        
        mock_client.return_value.project = "test-project"
        
        with patch.object(ModelMonitor, '_ensure_metrics_tables_exist'):
            monitor = ModelMonitor(project_id="test-project")
        
        decay_results = {'decay_detected': False}
        drift_results = {'overall_drift_detected': False}
        
        with patch.object(monitor, '_send_alert_notification') as mock_alert:
            result = monitor.create_alert_if_needed(
                decay_results=decay_results,
                drift_results=drift_results
            )
        
        assert result == False
        mock_alert.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
